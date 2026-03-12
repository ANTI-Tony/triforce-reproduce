# Autoregressive Baseline - 使用 tree.benchmark
import torch
import time
import argparse
import csv
import os
import sys

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from tree.benchmark import Benchmark

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from data_loader import load_prompts

CONTEXT_PRESETS = {
    'long':  124928,  # 122K
    'short': 3800,
}


class LlamaBenchmark(Benchmark):
    def __init__(self, log_dir):
        super().__init__(log_dir=log_dir)

    def initialize(self, model_dir, token_dir, **kwargs):
        """初始化 tokenizer 和 model"""
        self.tokenizer = AutoTokenizer.from_pretrained(token_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map='auto',
            attn_implementation="flash_attention_2"
        )
        self.model.eval()

    def generate_text(self, prompts, max_new_tokens=256, batch_size=1, prefill_chunk=1024):
        """AR generation with chunked prefill to avoid OOM on long contexts."""
        total_new_tokens = 0
        times = []
        eos_id = self.tokenizer.eos_token_id

        for i in tqdm(range(0, len(prompts), batch_size), desc="AR Generating"):
            text = prompts[i]
            input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.model.device)
            prompt_len = input_ids.shape[1]

            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                # Chunked prefill: process prompt in small chunks
                past = None
                last_logits = None
                for start in range(0, prompt_len, prefill_chunk):
                    end = min(start + prefill_chunk, prompt_len)
                    out = self.model(
                        input_ids=input_ids[:, start:end],
                        past_key_values=past,
                        use_cache=True
                    )
                    past = out.past_key_values
                    last_logits = out.logits[:, -1, :]
                    del out
                    if start % (prefill_chunk * 16) == 0:
                        torch.cuda.empty_cache()

                # First new token
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
                del last_logits
                new_tokens = 1

                # Autoregressive decode
                for _ in range(max_new_tokens - 1):
                    out = self.model(
                        input_ids=next_token,
                        past_key_values=past,
                        use_cache=True
                    )
                    past = out.past_key_values
                    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                    del out
                    new_tokens += 1
                    if next_token.item() == eos_id:
                        break

            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
            total_new_tokens += new_tokens
            # Free KV cache between samples
            del past
            torch.cuda.empty_cache()

        total_time = sum(times)
        throughput = total_new_tokens / total_time if total_time > 0 else 0

        print(f"Total New Tokens: {total_new_tokens}")
        print(f"Total Generation Time: {total_time:.2f} seconds")
        print(f"Throughput: {throughput:.2f} tokens/s")

        return total_new_tokens, total_time, throughput


def parse_args():
    parser = argparse.ArgumentParser(description="AR Baseline Benchmark")
    parser.add_argument("--context", type=str, required=True,
                        choices=["long", "short"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["gs", "longbench_packed_qmsum", "lwm"])
    parser.add_argument("--model_dir", type=str,
                        default="NousResearch/Yarn-Llama-2-7b-128k")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    prompt_len = CONTEXT_PRESETS[args.context]
    print(f"=== AR Baseline ===")
    print(f"Context: {args.context} (prompt={prompt_len})")
    print(f"Dataset: {args.dataset}")
    print(f"Model:   {args.model_dir}")
    print()

    worker = LlamaBenchmark(log_dir='ar_benchmark')
    worker.initialize(model_dir=args.model_dir, token_dir=args.model_dir)

    # 用 data_loader 加载数据集，将 token IDs 解码回文本（保证截断精确）
    prompts_data = load_prompts(args.dataset, worker.tokenizer, prompt_len, args.max_samples)
    prompt_texts = []
    for p in prompts_data:
        text = worker.tokenizer.decode(p['tokens'], skip_special_tokens=True)
        prompt_texts.append(text)
    print(f"Loaded {len(prompt_texts)} prompts\n")

    torch.cuda.reset_peak_memory_stats()

    # warmup
    print("=== Warmup ===")
    worker.generate_text(prompt_texts[:args.warmup], max_new_tokens=args.max_new_tokens)

    # benchmark
    print(f"\n=== Benchmark ({len(prompt_texts)} samples) ===")
    total_new_tokens, total_time, throughput = worker.generate_text(
        prompt_texts, max_new_tokens=args.max_new_tokens
    )

    ms_per_token = (total_time / total_new_tokens * 1000) if total_new_tokens > 0 else 0
    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"\n{'='*50}")
    print(f"Context:     {args.context}")
    print(f"Dataset:     {args.dataset}")
    print(f"ms/token:    {ms_per_token:.2f}")
    print(f"tokens/s:    {throughput:.2f}")
    print(f"Peak GPU:    {peak_gpu_mb:.0f} MB")
    print(f"{'='*50}")

    if args.output_csv:
        file_exists = os.path.exists(args.output_csv)
        with open(args.output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['context', 'dataset', 'method', 'ms_per_token',
                                 'throughput', 'total_tokens', 'total_time', 'peak_gpu_mb'])
            writer.writerow([
                args.context, args.dataset, 'ar_baseline',
                f"{ms_per_token:.2f}", f"{throughput:.2f}",
                total_new_tokens, f"{total_time:.2f}", f"{peak_gpu_mb:.0f}"
            ])
        print(f"Results appended to {args.output_csv}")
