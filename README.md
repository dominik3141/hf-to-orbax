# HF Safetensors → Orbax (TensorStore) with Smart Stacking

Convert Hugging Face `.safetensors` checkpoints into a JAX/Orbax TensorStore
checkpoint with stacked layer parameters (the layer index is removed, otherwise
the key name is preserved).
The converter is CPU-only and processes one stack at a time to keep memory use
manageable.

## Requirements

- GCS write access (ADC or service account)
- `gs://...` output path (unless `--local` is used)

Install dependencies (uv):

```bash
uv sync
```

## Usage

```bash
uv run hf-safetensors-to-orbax \
  --hf-repo google/gemma-3-27b \
  --output gs://my-bucket/gemma-3-27b-orbax
```

Local output (skips GCS):

```bash
uv run hf-safetensors-to-orbax \
  --hf-repo google/gemma-3-27b \
  --output ./orbax-out \
  --local
```

Optional token (for gated models):

```bash
uv run hf-safetensors-to-orbax \
  --hf-repo google/gemma-3-27b \
  --hf-token $HF_TOKEN \
  --output gs://my-bucket/gemma-3-27b-orbax
```

## What it does

- Downloads the safetensors snapshot from Hugging Face.
- Detects layer parameters by pattern (e.g. `model.layers.12.*`).
- Groups parameters by suffix and stacks across layers on axis 0.
- Writes a single Orbax checkpoint (unsharded/global arrays).

### Output naming

- Stacked: drop the numeric layer index, keep the rest of the key unchanged.
  - Example: `model.layers.12.self_attn.q_proj.weight` → `model.layers.self_attn.q_proj.weight`
- Global params: unchanged from Hugging Face (no prefix stripping or dot conversion).

## Notes

- The converter fails fast on missing layer indices or shape mismatches.
- Expect one stacked tensor at a time in RAM; a single stack for large models
  can be ~2GB.
- GCS auth must be configured before running (ADC or service account).
- Byte-level download progress uses `hf_transfer`, which is installed by default.
- Use `--local` to write to a filesystem path instead of GCS.
