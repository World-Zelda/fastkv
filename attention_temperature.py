import os
import json
import pandas as pd
import argparse
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


stop_words = ["```python", "```py", "Python code", "# Python", "import "]
# ===== 评分函数 =====
def normalize_answer(s):
    return re.sub(r'\s+', ' ', s.strip())

def extract_boxed_answer(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1] if matches else None

def compute_score(model_output, target_answer):
    boxed = extract_boxed_answer(model_output)
    if boxed is None:
        return 0, None, None
    extracted = normalize_answer(boxed)
    target = normalize_answer(target_answer)
    score = 1 if extracted == target else 0
    return score, boxed, extracted


# ===== 后处理函数 =====
def truncate_after_boxed(tokenizer, text, max_tokens_after_boxed=100):
    if "\\boxed" not in text:
        return text
    last_boxed_idx = text.rfind("\\boxed")
    prefix = text[:last_boxed_idx]
    suffix = text[last_boxed_idx:]
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    if len(suffix_tokens) > max_tokens_after_boxed:
        suffix_truncated = tokenizer.decode(suffix_tokens[:max_tokens_after_boxed])
        return prefix + suffix_truncated
    return text


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    prompts = [
        "Solve the following math problem step by step. "
        "Put your final answer in a boxed format at the end.\n\n"
        f"Question: {example['problem']}\n\n"
        for example in dataset
    ]

    llm = LLM(
        model=args.model_name,
        dtype="bfloat16",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_model_len=16384
    )

    # 构建 SamplingParams
    sampling_params = SamplingParams(
        n=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature if args.do_sample else 0.0,  # greedy when not sampling
        top_p=args.top_p if args.do_sample else 1.0,
        repetition_penalty=1.1,
        stop=stop_words,
        skip_special_tokens=True
    )

    # ===== 推理 =====
    print("Running inference with vLLM...")
    outputs = llm.generate(prompts, sampling_params)

    total = len(dataset)
    all_runs_scores = [[] for _ in range(args.num_samples)]
    all_runs_correct = [0] * args.num_samples

    for i, (example, output_group) in enumerate(zip(dataset, outputs)):
        assert len(output_group.outputs) == args.num_samples

        for sample_idx in range(args.num_samples):
            raw_output = output_group.outputs[sample_idx].text
            truncated_output = truncate_after_boxed(tokenizer, raw_output, max_tokens_after_boxed=100)
            score, boxed_content, extracted_answer = compute_score(truncated_output, example['answer'])

            all_runs_correct[sample_idx] += score
            all_runs_scores[sample_idx].append({
                'index': i,
                'problem': example['problem'],
                'model_output': truncated_output,
                'raw_output': raw_output,
                'boxed_content': boxed_content,
                'extracted_answer': extracted_answer,
                'target_answer': example['answer'],
                'score': score
            })

        if i < 3 and args.num_samples > 0:
            print(f"\n--- Example {i} Sample 0 ---")
            print(f"Problem: {example['problem'][:100]}...")
            print(f"Output: {truncate_after_boxed(tokenizer, output_group.outputs[0].text)[:200]}...")

    # ===== 保存结果 =====
    model_short_name = args.model_name.split("/")[-1]
    output_base_dir = os.path.join(args.output_dir, model_short_name, "MATH-500")
    os.makedirs(output_base_dir, exist_ok=True)  # 自动创建所有父目录

    for run_id in range(1, args.num_samples + 1):
        scores_list = all_runs_scores[run_id - 1]
        accuracy = all_runs_correct[run_id - 1] / total

        # 文件名不含路径前缀，只用 runX
        filename_base = os.path.join(output_base_dir, f"run{run_id}")

        save_data = {
            'model_name': args.model_name,
            'dataset': "MATH-500",
            'do_sample': args.do_sample,
            'num_samples': args.num_samples,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'run_id': run_id,
            'total_examples': total,
            'accuracy': accuracy,
            'scores': scores_list
        }

        # 保存 JSON
        with open(filename_base + ".json", 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        # 保存 CSV
        df = pd.DataFrame([{
            'index': item['index'],
            'score': item['score'],
            'extracted_answer': item['extracted_answer'],
            'target_answer': item['target_answer'],
            'has_boxed': 1 if item['boxed_content'] else 0
        } for item in scores_list])
        df.to_csv(filename_base + ".csv", index=False, encoding='utf-8')

        print(f"Run {run_id}: Accuracy = {accuracy:.2%} → Saved to {filename_base}.json/.csv")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    # ===== 环境 & 初始化 =====
    parser = argparse.ArgumentParser(description="Evaluate math reasoning with vLLM and multiple sampling.")
    parser.add_argument("--model_name", type=str, required=True, help="Path or HuggingFace model name")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling (otherwise greedy decoding)")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per prompt (n). Ignored if do_sample=False.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for sampling")
    parser.add_argument("--max_tokens", type=int, default=12000, help="Max tokens to generate")
    parser.add_argument("--output_dir", type=str, default="", help="Optional prefix for output files")
    args = parser.parse_args()

    # 强制：如果不采样，则 num_samples = 1
    if not args.do_sample:
        args.num_samples = 1

    print(f"Model: {args.model_name}")
    print(f"Do sample: {args.do_sample}")
    print(f"Num samples: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    main(args)