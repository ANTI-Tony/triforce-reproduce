#!/usr/bin/env bash
# EAGLE-3 evaluation: 4K/8K/16K/32K × 3 datasets
# Using LLaMA-3.1-8B-Instruct + EAGLE3 checkpoint
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESULTS_DIR="results/eagle3"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  EAGLE-3 Evaluation"
echo "  Base: LLaMA-3.1-8B-Instruct"
echo "  Draft: EAGLE3-LLaMA3.1-Instruct-8B"
echo "  4K/8K/16K/32K × gs/longbench/lwm"
echo "========================================="

# Install EAGLE if not already
if [ ! -d "/workspace/EAGLE" ]; then
    cd /workspace && git clone https://github.com/SafeAILab/EAGLE.git
    cd /workspace/EAGLE && pip install -e . -q
fi

cd /workspace/EAGLE

for PROMPT_LEN in 3800 8192 16384 32768; do
  for DS in gs longbench_packed_qmsum lwm; do
    echo ""
    echo "--- EAGLE-3: prompt=${PROMPT_LEN}, dataset=${DS} ---"

    python3 -c "
import torch
import time
import json
import sys
sys.path.insert(0, '/workspace/triforce-experiment')

from eagle.model.ea_model import EaModel

# Load model
model = EaModel.from_pretrained(
    use_eagle3=True,
    base_model_path='meta-llama/Llama-3.1-8B-Instruct',
    ea_model_path='yuhuili/EAGLE3-LLaMA3.1-Instruct-8B',
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='auto',
    total_token=60,
    depth=5,
    top_k=10,
)
model.eval()
tokenizer = model.tokenizer

# Load dataset
from sd_code.hl.eval_tinydraft import load_prompts
prompts = load_prompts('${DS}', tokenizer, max_length=${PROMPT_LEN}, max_samples=11)

print(f'Loaded {len(prompts)} prompts')

# Warmup
input_ids = prompts[0].unsqueeze(0).cuda()
_ = model.eagenerate(input_ids, temperature=0.0, max_new_tokens=32)

# Evaluate
results = []
for i, p in enumerate(prompts[1:]):  # skip warmup
    input_ids = p.unsqueeze(0).cuda()
    prompt_len = input_ids.shape[1]

    torch.cuda.synchronize()
    t0 = time.time()
    output = model.eagenerate(input_ids, temperature=0.0, max_new_tokens=256)
    torch.cuda.synchronize()
    t1 = time.time()

    gen_tokens = output.shape[1] - prompt_len
    throughput = gen_tokens / (t1 - t0)
    print(f'  sample {i+1}: prompt={prompt_len}, gen={gen_tokens}, {throughput:.1f} tok/s, {t1-t0:.1f}s')
    results.append({'throughput': throughput, 'gen_tokens': gen_tokens, 'time': t1-t0})

avg_throughput = sum(r['throughput'] for r in results) / len(results)
print(f'Average throughput: {avg_throughput:.2f} tok/s')

# Save
with open('${RESULTS_DIR}/eagle3_${PROMPT_LEN}_${DS}.json', 'w') as f:
    json.dump({'prompt_len': ${PROMPT_LEN}, 'dataset': '${DS}', 'avg_throughput': avg_throughput, 'results': results}, f, indent=2)
" 2>&1 | tee "${RESULTS_DIR}/eagle3_${PROMPT_LEN}_${DS}.log"

  done
done

echo ""
echo "========================================="
echo "  EAGLE-3 All Done!"
echo "========================================="
