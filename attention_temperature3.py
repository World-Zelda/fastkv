# run_math_samples.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import pandas as pd
import argparse
from datasets import load_dataset
from vllm import LLM, SamplingParams
from utils.attention_temperature_utils import compute_score

stop_words = ["```python", "```py", "Python code", "# Python", "import "]

# 根据数据集名称自动选择 split 和字段名
dataset_config = {
    "HuggingFaceH4/aime_2024": {"split": "train", "question_key": "problem"},
    "opencompass/AIME2025": {"split": "test", "question_key": "question"},  # 注意：你设为 test
    "FlagEval/HMMT_2025": {"split": "train", "question_key": "question"},
    "HuggingFaceH4/MATH-500": {"split": "test", "question_key": "problem"},
}

# 特殊处理 AIME2025 的 configs
AIME2025_CONFIGS = ["AIME2025-I", "AIME2025-II"]

def main(args):
    if args.dataset not in dataset_config:
        raise ValueError(f"Dataset {args.dataset} not configured in dataset_config!")

    config_meta = dataset_config[args.dataset]
    split = config_meta["split"]
    question_key = config_meta["question_key"]

    model_name_clean = args.model_name.split("/")[-1]
    base_dataset_clean = args.dataset.split("/")[-1]  # e.g., "AIME2025"

    # 初始化模型（只初始化一次）
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

    # 决定要处理的数据集列表：(config_name, dataset_obj)
    if args.dataset == "opencompass/AIME2025":
        dataset_items = []
        for cfg in AIME2025_CONFIGS:
            print(f"Loading config: {cfg}")
            ds = load_dataset(args.dataset, cfg, split=split)
            dataset_items.append((cfg, ds))
    else:
        # 普通数据集
        ds = load_dataset(args.dataset, split=split)
        dataset_items = [(None, ds)]

    # 遍历每个 config（或普通数据集）
    for config_name, full_dataset in dataset_items:
        total_examples = len(full_dataset)
        print(f"\nProcessing dataset: {args.dataset}, config: {config_name or 'N/A'}, samples: {total_examples}")

        # 构建输出目录：.../AIME2025/AIME2025-I/0.8/
        output_dir_parts = [
            args.output_base_dir,
            model_name_clean,
            base_dataset_clean
        ]
        if config_name:
            output_dir_parts.append(config_name)  # 插入 config 名
        output_dir_parts.append(str(args.temperature))
        output_dir = os.path.join(*output_dir_parts)
        os.makedirs(output_dir, exist_ok=True)

        # 为每个 run 准备 JSONL 文件
        run_files = []
        run_writers = []
        for run_id in range(args.num_samples):
            run_path = os.path.join(output_dir, f"run{run_id + 1}_samples.jsonl")
            f = open(run_path, 'w', encoding='utf-8')
            run_writers.append(f)
        total_correct = [0] * args.num_samples

        # 逐条推理
        for idx, example in enumerate(full_dataset):
            print(f"\n[Sample {idx + 1}/{total_examples}] Generating...")

            prompt = (
                "Solve the following math problem step by step. "
                "Put your final answer in a boxed format at the end.\n\n"
                f"Question: {example[question_key]}\n\n"
            )

            outputs = llm.generate([prompt], sampling_params)
            output = outputs[0]

            for run_id in range(args.num_samples):
                raw_output = output.outputs[run_id].text
                score, boxed, extracted = compute_score(raw_output, example['answer'])
                total_correct[run_id] += score

                record = {
                    'global_index': idx,
                    'problem': example[question_key],
                    'boxed_content': boxed,
                    'extracted_answer': extracted,
                    'target_answer': example['answer'],
                    'score': score,
                    'raw_output': raw_output,
                }

                run_writers[run_id].write(json.dumps(record, ensure_ascii=False) + "\n")
                run_writers[run_id].flush()

        # 关闭文件
        for f in run_writers:
            f.close()

        # 汇总生成最终 .json 和 .csv
        print("\n Aggregating results and generating CSVs...")
        for run_id in range(args.num_samples):
            run_path = os.path.join(output_dir, f"run{run_id + 1}_samples.jsonl")
            records = []
            with open(run_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))

            acc = total_correct[run_id] / total_examples
            meta_path = os.path.join(output_dir, f"run{run_id + 1}.json")
            meta = {
                'model_name': args.model_name,
                'dataset': args.dataset,
                'config': config_name,
                'split': split,
                'do_sample': args.do_sample,
                'num_samples': args.num_samples,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'run_id': run_id + 1,
                'num_examples': total_examples,
                'accuracy': acc,
                'scores': records
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            df = pd.DataFrame([{
                'global_index': r['global_index'],
                'score': r['score'],
                'extracted_answer': r['extracted_answer'],
                'target_answer': r['target_answer'],
                'has_boxed': 1 if r['boxed_content'] else 0
            } for r in records])
            df.to_csv(os.path.join(output_dir, f"run{run_id + 1}.csv"), index=False, encoding='utf-8')

            print(f"✅ Run {run_id + 1}: Acc={acc:.2%} → Saved to {meta_path}")

    print(f"\n🎉 All done! Results saved under: {args.output_base_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500",
                        help="Hugging Face dataset name")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=30000)
    parser.add_argument("--output_base_dir", type=str, default="./results")
    args = parser.parse_args()

    if not args.do_sample:
        args.num_samples = 1

    main(args)