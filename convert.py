#!/usr/bin/env python3
"""Convert Hugging Face safetensors to Orbax (TensorStore) with smart stacking.

This program streams a Hugging Face safetensors snapshot, discovers transformer
layer parameters, stacks them by suffix across depth, and writes a topology-
agnostic Orbax checkpoint to GCS. It enforces CPU execution, processes one stack
at a time to limit peak RAM, and keeps non-layer parameters unstacked with clean
names for downstream JAX/Orbax training loops.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Force CPU before importing JAX.
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import typer
from huggingface_hub import snapshot_download
from safetensors import safe_open
from tqdm import tqdm

import orbax.checkpoint as ocp


APP_NAME = "hf-safetensors-to-orbax"
ALLOWED_LAYER_PREFIXES = {"layers", "layer", "h", "blocks", "block"}
PREFIXES_TO_STRIP = ("model.", "transformer.", "module.")
WEIGHT_SUFFIX = ".weight"


def ensure_gcs_path(path: str) -> str:
    """Validate and normalize the output location as a GCS path.

    The converter is intentionally GCS-only to avoid confusing local paths or
    partial outputs. This helper enforces the scheme, trims trailing slashes,
    and fails fast when the input is clearly not a usable GCS prefix.
    """
    if not path.startswith("gs://") or len(path) <= 5:
        raise typer.BadParameter("--gcs-bucket must be a valid gs:// path")
    return path.rstrip("/")


def configure_logging() -> None:
    """Configure structured logging for CLI execution.

    The goal is to surface high-level progress milestones and critical context
    (like stack shapes and layer ranges) without drowning the user in per-tensor
    noise. The format is timestamped for long-running conversions.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def download_snapshot(repo_id: str, token: Optional[str]) -> str:
    """Download the safetensors snapshot from Hugging Face.

    Uses the Hub snapshot API to materialize only safetensors and their optional
    index JSON. This isolates the conversion to weight files and avoids pulling
    large artifacts that are irrelevant to the conversion pipeline.
    """
    logging.info("Downloading snapshot for %s", repo_id)
    return snapshot_download(
        repo_id=repo_id,
        token=token,
        allow_patterns=["*.safetensors", "*.safetensors.index.json"],
    )


def find_index_json(snapshot_dir: str) -> Optional[str]:
    """Locate a safetensors index JSON, if present.

    Sharded HF checkpoints publish a single index JSON that maps tensor names
    to shard files. We treat multiple index files as an error because it is not
    a standard layout and likely indicates a mixed or malformed snapshot.
    """
    matches = list(Path(snapshot_dir).rglob("*.safetensors.index.json"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError("Multiple safetensors index JSON files found")
    return str(matches[0])


def build_key_to_file(snapshot_dir: str) -> Dict[str, str]:
    """Build a mapping from tensor key to safetensors file path.

    Require the official shard index to avoid redundant scans and to make the
    conversion deterministic. Duplicate keys are treated as fatal to prevent
    silent corruption.
    """
    index_path = find_index_json(snapshot_dir)
    if not index_path:
        raise ValueError("Missing safetensors index JSON file")

    with open(index_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    weight_map = data.get("weight_map")
    if not weight_map:
        raise ValueError(f"Index JSON missing weight_map: {index_path}")
    mapping: Dict[str, str] = {}
    for key, rel_path in weight_map.items():
        abs_path = os.path.join(snapshot_dir, rel_path)
        if key in mapping:
            raise ValueError(f"Duplicate key in index: {key}")
        mapping[key] = abs_path
    return mapping


def parse_layer_key(key: str) -> Optional[Tuple[int, str]]:
    """Parse a parameter key into (layer_index, suffix) if it looks stackable.

    The heuristic looks for an integer segment that is preceded by a known layer
    container name (layers, h, blocks, ...). The suffix is everything after the
    index, which becomes the grouping key for stacking.
    """
    parts = key.split(".")
    for idx in range(1, len(parts) - 1):
        if parts[idx].isdigit() and parts[idx - 1] in ALLOWED_LAYER_PREFIXES:
            suffix = ".".join(parts[idx + 1 :]).strip(".")
            if suffix:
                return int(parts[idx]), suffix
    return None


def group_layer_keys(keys: Iterable[str]) -> Tuple[Dict[str, Dict[int, str]], List[str]]:
    """Group layer parameters by suffix and separate non-layer parameters.

    This pass identifies all stackable keys, buckets them by suffix, and filters
    the remaining parameters into a non-layer list. Only suffix groups with two
    or more distinct layer indices qualify for stacking to reduce false positives.
    """
    suffix_groups: Dict[str, Dict[int, str]] = {}

    for key in keys:
        parsed = parse_layer_key(key)
        if not parsed:
            continue
        layer_idx, suffix = parsed
        group = suffix_groups.setdefault(suffix, {})
        if layer_idx in group:
            raise ValueError(f"Duplicate layer index {layer_idx} for suffix {suffix}")
        group[layer_idx] = key

    stackable_suffixes = {
        suffix: group for suffix, group in suffix_groups.items() if len(group) >= 2
    }

    stackable_keys = {key for group in stackable_suffixes.values() for key in group.values()}
    non_layer_keys = [key for key in keys if key not in stackable_keys]

    return stackable_suffixes, non_layer_keys


def load_tensor(file_path: str, key: str) -> np.ndarray:
    """Load a single tensor from a safetensors file.

    Safetensors reads are isolated to the specific key to keep IO and memory
    tight. Callers handle validation and stacking; this only returns the array.
    """
    with safe_open(file_path, framework="numpy") as handle:
        return handle.get_tensor(key)


def validate_contiguous(indices: List[int], suffix: str) -> None:
    """Assert that a suffix group has a contiguous layer index range.

    Stacking assumes a dense layer axis. Missing indices likely mean a keying
    bug or an unexpected model layout, so we fail fast with a helpful message.
    """
    expected = list(range(indices[0], indices[-1] + 1))
    if indices != expected:
        raise ValueError(
            f"Missing layer indices for suffix '{suffix}': got {indices}, expected {expected}"
        )


def make_stacked_name(suffix: str) -> str:
    """Translate a suffix into the output stacked parameter name.

    The suffix path is preserved while converting dots to slashes, and it is
    rooted under `layers_stacked/` to make stacked arrays explicit and distinct
    from non-layer parameters in the output tree.
    """
    return "layers_stacked/" + suffix.replace(".", "/")


def clean_global_name(key: str) -> str:
    """Normalize a non-layer parameter key into a clean output name.

    This strips common framework prefixes and a trailing `.weight`, then
    converts dot segments into path separators for a stable, readable tree.
    """
    cleaned = key
    for prefix in PREFIXES_TO_STRIP:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    if cleaned.endswith(WEIGHT_SUFFIX):
        cleaned = cleaned[: -len(WEIGHT_SUFFIX)]
    cleaned = cleaned.strip(".")
    if not cleaned:
        raise ValueError(f"Invalid empty name after cleaning: {key}")
    return cleaned.replace(".", "/")


def save_array_to_temp(temp_dir: str, name: str, array: np.ndarray) -> np.ndarray:
    """Persist an array to disk and reopen it as a memory-mapped view.

    Building the full output tree in RAM can be expensive for large models. By
    staging arrays to disk and reopening with mmap, we limit peak memory while
    still handing Orbax a stable array-like object for checkpoint writing.
    """
    file_path = os.path.join(temp_dir, f"{name}.npy")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, array)
    return np.load(file_path, mmap_mode="r")


def convert(
    hf_repo: str,
    hf_token: Optional[str],
    gcs_bucket: str,
    stacking_config: str,
) -> None:
    """Run the full conversion pipeline from HF snapshot to Orbax checkpoint.

    The flow is: download snapshot, build key-to-file map, identify stackable
    groups, stack one suffix group at a time with shape validation, write staged
    arrays into an output tree, then save via Orbax. This function enforces all
    invariants and ensures temporary storage is cleaned up on failure.
    """
    if stacking_config != "auto":
        raise typer.BadParameter("--stacking-config only supports 'auto'")

    gcs_bucket = ensure_gcs_path(gcs_bucket)
    snapshot_dir = download_snapshot(hf_repo, hf_token)
    key_to_file = build_key_to_file(snapshot_dir)

    keys = sorted(key_to_file.keys())
    stackable, non_layer_keys = group_layer_keys(keys)

    logging.info("Found %d stackable groups", len(stackable))
    logging.info("Found %d non-layer params", len(non_layer_keys))

    output_tree: Dict[str, np.ndarray] = {}
    used_names = set()

    temp_dir = tempfile.mkdtemp(prefix=f"{APP_NAME}-")
    logging.info("Using temp dir: %s", temp_dir)

    try:
        for suffix in tqdm(sorted(stackable.keys()), desc="Stacking groups"):
            group = stackable[suffix]
            indices = sorted(group.keys())
            validate_contiguous(indices, suffix)

            first_key = group[indices[0]]
            first_arr = load_tensor(key_to_file[first_key], first_key)
            stacked = np.empty((len(indices),) + first_arr.shape, dtype=first_arr.dtype)
            stacked[0] = first_arr
            base_shape = first_arr.shape
            logging.info(
                "Stacking %s | layers %d-%d | shape %s | dtype %s",
                suffix,
                indices[0],
                indices[-1],
                base_shape,
                stacked.dtype,
            )
            del first_arr

            for pos, layer_idx in enumerate(indices[1:], start=1):
                key = group[layer_idx]
                arr = load_tensor(key_to_file[key], key)
                if arr.shape != base_shape:
                    raise ValueError(
                        f"Shape mismatch for suffix '{suffix}': {arr.shape} vs {base_shape}"
                    )
                stacked[pos] = arr

            stacked_name = make_stacked_name(suffix)
            if stacked_name in used_names:
                raise ValueError(f"Output name collision: {stacked_name}")
            used_names.add(stacked_name)

            output_tree[stacked_name] = save_array_to_temp(
                temp_dir, stacked_name, stacked
            )
            del stacked
            gc.collect()

        for key in tqdm(non_layer_keys, desc="Saving non-layer params"):
            global_name = clean_global_name(key)
            if global_name in used_names:
                raise ValueError(f"Output name collision: {global_name}")
            used_names.add(global_name)

            arr = load_tensor(key_to_file[key], key)
            output_tree[global_name] = save_array_to_temp(temp_dir, global_name, arr)
            del arr

        logging.info("Saving Orbax checkpoint to %s", gcs_bucket)
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(gcs_bucket, output_tree)
        logging.info("Save complete")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    hf_repo: str = typer.Option(..., help="Hugging Face repo ID"),
    hf_token: Optional[str] = typer.Option(None, help="Hugging Face auth token"),
    gcs_bucket: str = typer.Option(..., help="Target GCS path (gs://...)"),
    stacking_config: str = typer.Option("auto", help="Stacking strategy (only 'auto')"),
) -> None:
    """CLI entrypoint that wires arguments to the conversion routine.

    This keeps the CLI surface thin while centralizing the conversion logic in
    `convert`, which makes it easier to test or reuse programmatically.
    """

    configure_logging()
    convert(hf_repo=hf_repo, hf_token=hf_token, gcs_bucket=gcs_bucket, stacking_config=stacking_config)


if __name__ == "__main__":
    try:
        app()
    except Exception as exc:  # pragma: no cover - CLI failure handling
        logging.error("Conversion failed: %s", exc)
        sys.exit(1)
