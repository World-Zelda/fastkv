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
    total_examples = len(dataset)

    # 验证范围
    assert 0 <= args.start < total_examples, f"start must be in [0, {total_examples})"
    assert args.start < args.end <= total_examples, f"end must be in (start, {total_examples}]"
    
    selected_indices = list(range(args.start, args.end))
    subset = dataset.select(selected_indices)
    print(f"Processing examples [{args.start}, {args.end}) → {len(subset)} samples")

    # 构建 prompts
    prompts = [
        "Solve the following math problem step by step. "
        "Put your final answer in a boxed format at the end.\n\n"
        f"Question: {ex['problem']}\n\n"
        for ex in subset
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

    print("Running inference...")
    outputs = llm.generate(prompts, sampling_params)

    # 为每个 run 准备容器
    runs_data = [[] for _ in range(args.num_samples)]
    runs_correct = [0] * args.num_samples

    for i, output in enumerate(outputs):
        original_idx = selected_indices[i]
        example = subset[i]
        for run_id in range(args.num_samples):
            raw_output = output.outputs[run_id].text
            score, boxed, extracted = compute_score(raw_output, example['answer'])
            runs_correct[run_id] += score
            runs_data[run_id].append({
                'global_index': original_idx,  # 记录原始全局 index
                'local_index': i,
                'problem': example['problem'],
                'raw_output': raw_output,
                'boxed_content': boxed,
                'extracted_answer': extracted,
                'target_answer': example['answer'],
                'score': score
            })

    # 保存路径：temperature/start-end/
    model_name_clean = args.model_name.split("/")[-1]
    segment_dir = f"{args.start}-{args.end}"
    output_dir = os.path.join(
        args.output_base_dir,
        model_name_clean,
        "MATH-500",
        str(args.temperature),
        segment_dir
    )
    os.makedirs(output_dir, exist_ok=True)

    # 保存每个 run
    for run_id in range(args.num_samples):
        data = runs_data[run_id]
        acc = runs_correct[run_id] / len(subset)
        base_path = os.path.join(output_dir, f"run{run_id + 1}")

        meta = {
            'model_name': args.model_name,
            'dataset': "MATH-500",
            'do_sample': args.do_sample,
            'num_samples': args.num_samples,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'run_id': run_id + 1,
            'segment': segment_dir,
            'global_start': args.start,
            'global_end': args.end,
            'num_examples_in_segment': len(subset),
            'accuracy_in_segment': acc,
            'scores': data
        }

        with open(base_path + ".json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        df = pd.DataFrame([{
            'global_index': item['global_index'],
            'score': item['score'],
            'extracted_answer': item['extracted_answer'],
            'target_answer': item['target_answer'],
            'has_boxed': 1 if item['boxed_content'] else 0
        } for item in data])
        df.to_csv(base_path + ".csv", index=False, encoding='utf-8')

        print(f"✅ Run {run_id + 1}: Acc={acc:.2%} → Saved to {base_path}.json")

    print(f"\n🎉 Segment [{args.start}, {args.end}) completed.")


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
    parser.add_argument("--output_base_dir", type=str, default="./outputs")
    args = parser.parse_args()

    if not args.do_sample:
        args.num_samples = 1

    main(args)