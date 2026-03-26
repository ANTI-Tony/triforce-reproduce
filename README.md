# TriForce Reproduction Experiment

Reproduce the TriForce paper (arXiv:2404.11912) experiments as a baseline.
TriForce is a hierarchical KV cache-based speculative decoding method for accelerating long-context LLM inference.

## Setup

**Target hardware:** RunPod A100 80GB

### Quick Start (RunPod Jupyter)

1. Upload or clone this repo to RunPod
2. Open `notebooks/TriForce_RunPod_A100.ipynb`
3. Run cells sequentially

### Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install flash-attn==2.5.7 --no-build-isolation

# 2. Clone TriForce
bash scripts/clone_triforce.sh

# 3. Prepare data
python scripts/prepare_data.py

# 4. Download models
bash scripts/download_models.sh
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Target Model | NousResearch/Yarn-Llama-2-7b-128k | 128K context LLaMA |
| Draft Model | JackFram/llama-68m | 68M parameter small model |
| prefill | 124928 (~122K tokens) | Official default |
| gen_len | 256 | Generated token count |
| budget | 4096 | Retrieval cache budget |
| chunk_size | 8 | Retrieval granularity |
| draft_cache_budget | 256 | StreamingLLM cache |
| gamma | 6 | Speculation length |

## Expected Results (from paper)

| Method | Latency (ms/token) | Speedup |
|--------|-------------------|---------|
| Autoregressive | ~46-48 | 1.0x |
| TriForce | ~21-22 | ~2.1-2.2x |

Average acceptance rate: ~0.85-0.90

## Memory Estimate

- Model weights: ~14GB
- KV cache: ~32GB
- Other: ~10GB
- **Total: ~56GB** (fits A100 80GB)

## References

- Paper: https://arxiv.org/abs/2404.11912
- Code: https://github.com/Infini-AI-Lab/TriForce
