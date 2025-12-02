# aggregate_and_score.py
import os
import json
import argparse
import pandas as pd
from glob import glob

def main(args):
    model_name_clean = args.model_name.split("/")[-1]
    temp_dir = os.path.join(
        args.output_base_dir,
        model_name_clean,
        args.dataset,
        str(args.temperature)
    )

    if not os.path.exists(temp_dir):
        raise ValueError(f"Directory not found: {temp_dir}")

    # 找到所有 segment 目录（如 0-500, 500-1000）
    segment_dirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
    segment_dirs.sort(key=lambda x: int(x.split('-')[0]))  # 按起始 index 排序

    print(f"Found segments: {segment_dirs}")

    # 获取 run 数量（从任意 segment 中读取）
    sample_segment = os.path.join(temp_dir, segment_dirs[0])
    run_files = sorted(glob(os.path.join(sample_segment, "run*.json")))
    num_runs = len(run_files)
    print(f"Detected {num_runs} runs.")

    # 为每个 run 收集所有 segment 的结果
    all_runs_data = [[] for _ in range(num_runs)]
    total_examples = 0

    for seg in segment_dirs:
        seg_path = os.path.join(temp_dir, seg)
        for run_id in range(num_runs):
            json_path = os.path.join(seg_path, f"run{run_id + 1}.json")
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Missing file: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_runs_data[run_id].extend(data['scores'])
        # 假设所有 segment 的 example 数一致？不，我们累加
        # 用第一个 run 的 segment 信息获取数量
        with open(os.path.join(seg_path, "run1.json"), 'r') as f:
            seg_meta = json.load(f)
            total_examples += seg_meta['num_examples_in_segment']

    print(f"Total examples aggregated: {total_examples}")

    # 计算每个 run 的总准确率
    results = []
    for run_id in range(num_runs):
        scores = all_runs_data[run_id]
        assert len(scores) == total_examples, f"Run {run_id+1} has {len(scores)} != {total_examples}"
        correct = sum(item['score'] for item in scores)
        acc = correct / total_examples
        results.append({
            'run_id': run_id + 1,
            'correct': correct,
            'total': total_examples,
            'accuracy': acc
        })
        print(f"📊 Run {run_id + 1}: {correct}/{total_examples} = {acc:.2%}")

    # 保存汇总结果
    summary_path = os.path.join(temp_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': args.model_name,
            'temperature': args.temperature,
            'num_runs': num_runs,
            'total_examples': total_examples,
            'per_run_results': results
        }, f, indent=2, ensure_ascii=False)

    # 也保存一个 CSV
    df_summary = pd.DataFrame(results)
    df_summary.to_csv(os.path.join(temp_dir, "summary.csv"), index=False)

    print(f"\n✅ Summary saved to {summary_path} and summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--output_base_dir", type=str, default="./outputs")
    parser.add_argument("--dataset", type=str, default="MATH-500")
    args = parser.parse_args()

    main(args)