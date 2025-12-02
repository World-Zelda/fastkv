# run_math_samples.py
import os
import json
import pandas as pd
import argparse
from datasets import load_dataset
from vllm import LLM, SamplingParams
from utils.attention_temperature_utils import compute_score

stop_words = ["```python", "```py", "Python code", "# Python", "import "]


def main(args):
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    # 构建 prompts
    prompts = [
        "Solve the following math problem step by step. "
        "Put your final answer in a boxed format at the end.\n\n"
        f"Question: {ex['problem']}\n\n"
        for ex in dataset
    ]

    # 初始化模型
    llm = LLM(
        model=args.model_name,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=16384,
        enable_prefix_caching=True
    )

    sampling_params = SamplingParams(
        n=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature if args.do_sample else 0.0,
        top_p=args.top_p if args.do_sample else 1.0,
        repetition_penalty=1.1,
        stop=stop_words,
        skip_special_tokens=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--start", type=int, required=True, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End index (exclusive)")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=12000)
    args = parser.parse_args()

    if not args.do_sample:
        args.num_samples = 1

    main(args)