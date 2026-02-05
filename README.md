# HF Safetensors → Orbax (TensorStore) with Smart Stacking

Convert Hugging Face `.safetensors` checkpoints into a JAX/Orbax TensorStore
checkpoint with stacked layer parameters (e.g. `layers_stacked/self_attn/q_proj`).
The converter is CPU-only and processes one stack at a time to keep memory use
manageable.

## Requirements

- GCS write access (ADC or service account)
- `gs://...` output path (local paths are rejected)

Install dependencies (uv):

```bash
uv sync
```

## Usage

```bash
uv run hf-safetensors-to-orbax \
  --hf-repo google/gemma-3-27b \
  --gcs-bucket gs://my-bucket/gemma-3-27b-orbax
```

Local output (skips GCS):

```bash
uv run hf-safetensors-to-orbax \
  --hf-repo google/gemma-3-27b \
  --gcs-bucket ./orbax-out \
  --local
```

Optional token (for gated models):

```bash
uv run hf-safetensors-to-orbax \
  --hf-repo google/gemma-3-27b \
  --hf-token $HF_TOKEN \
  --gcs-bucket gs://my-bucket/gemma-3-27b-orbax
```

## What it does

- Downloads the safetensors snapshot from Hugging Face.
- Detects layer parameters by pattern (e.g. `model.layers.12.*`).
- Groups parameters by suffix and stacks across layers on axis 0.
- Writes a single Orbax checkpoint (unsharded/global arrays).

### Output naming

- Stacked: `layers_stacked/<suffix path>`
  - Example: `model.layers.12.self_attn.q_proj` → `layers_stacked/self_attn/q_proj`
- Global params: stripped prefixes + `.weight`, then dots → `/`
  - Example: `model.embed_tokens.weight` → `embed_tokens`

## Notes

- The converter fails fast on missing layer indices or shape mismatches.
- Expect one stacked tensor at a time in RAM; a single stack for large models
  can be ~2GB.
- GCS auth must be configured before running (ADC or service account).
- Byte-level download progress uses `hf_transfer`, which is installed by default.
- Use `--local` to write to a filesystem path instead of GCS.
