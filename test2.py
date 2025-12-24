import os
import pandas as pd
import numpy as np
from math import comb

def pass_at_k(results: np.ndarray, k: int) -> float:
    n, m = results.shape
    if k > n:
        return np.nan
    sample_results = results.T
    total_pass = 0.0
    for i in range(m):
        successes = int(np.sum(sample_results[i]))
        failures = n - successes
        if successes >= k:
            total_pass += 1.0
        elif failures >= k:
            fail_comb = comb(failures, k)
            total_comb = comb(n, k)
            total_pass += 1.0 - (fail_comb / total_comb)
        else:
            total_pass += 1.0
    return total_pass / m

def load_csvs_from_dir(dir_path: str) -> np.ndarray:
    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    if not csv_files:
        raise ValueError("No CSV files")
    
    all_data = []
    for fname in sorted(csv_files):
        df = pd.read_csv(os.path.join(dir_path, fname))
        if 'global_index' not in df.columns or 'score' not in df.columns:
            raise ValueError(f"Missing columns in {fname}")
        df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0).astype(int)
        df = df[['global_index', 'score']].set_index('global_index')
        all_data.append(df)
    
    all_indices = sorted(set(idx for df in all_data for idx in df.index))
    aligned = [df.reindex(all_indices, fill_value=0)['score'].values for df in all_data]
    return np.array(aligned)

def extract_metadata_robust(leaf_dir: str):
    """
    从叶子目录路径中提取 model, dataset, temperature
    特殊处理 AIME2025-I / AIME2025-II
    """
    parts = os.path.normpath(leaf_dir).split(os.sep)
    
    # 初始化
    model = dataset = temp_str = "unknown"

    # 查找 AIME2025 特殊结构
    aime_handled = False
    for i in range(len(parts) - 1):
        if parts[i] == "AIME2025":
            if i + 1 < len(parts):
                next_part = parts[i + 1]
                if next_part in ("AIME2025-I", "AIME2025-II"):
                    # 找到子数据集
                    dataset = next_part
                    # model 是 AIME2025 的上一级
                    if i - 1 >= 0:
                        model = parts[i - 1]
                    # temperature 在子数据集之后
                    if i + 2 < len(parts):
                        temp_candidate = parts[i + 2]
                        # 判断是否为温度（可能是数字或 range）
                        # 如果下下级像 range（如 0-100），则再下一级才是温度？
                        # 但通常结构是: .../model/AIME2025/AIME2025-I/1.0/[0-100]/
                        # 所以温度在 i+2
                        temp_str = temp_candidate
                    else:
                        temp_str = "unknown"
                    aime_handled = True
                    break

    if not aime_handled:
        # 普通逻辑：尝试从末尾提取
        if len(parts) >= 4:
            last_part = parts[-1]
            # 判断最后一级是否为 range（含数字和 -）
            if '-' in last_part and any(c.isdigit() for c in last_part):
                # 可能是 range 目录，跳过
                if len(parts) >= 5:
                    model = parts[-4]
                    dataset = parts[-3]
                    temp_str = parts[-2]
                else:
                    model = parts[-3]
                    dataset = parts[-2]
                    temp_str = parts[-1]
            else:
                # 最后一级不是 range
                model = parts[-3]
                dataset = parts[-2]
                temp_str = parts[-1]
        elif len(parts) >= 3:
            model = parts[-3]
            dataset = parts[-2]
            temp_str = parts[-1]
        else:
            model = dataset = temp_str = "unknown"

    # 解析 temperature
    try:
        temperature = float(temp_str)
    except (ValueError, TypeError):
        temperature = temp_str

    return model, dataset, temperature

def main():
    root_dir = "results90"
    output_csv = "all_results90.csv"
    rows = []

    # 遍历所有目录，找出包含 .csv 的叶子目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        csv_files = [f for f in filenames if f.endswith('.csv')]
        if not csv_files:
            continue

        print(f"处理: {dirpath}")
        try:
            results = load_csvs_from_dir(dirpath)
            n_runs, n_samples = results.shape
            overall_acc = float(np.mean(results))

            model, dataset, temperature = extract_metadata_robust(dirpath)

            # 计算 pass@k
            pass1 = pass_at_k(results, 1)
            pass3 = pass_at_k(results, 3) if n_runs >= 3 else np.nan
            pass5 = pass_at_k(results, 5) if n_runs >= 5 else np.nan
            pass10 = pass_at_k(results, 10) if n_runs >= 10 else np.nan

            rows.append({
                "model": model,
                "dataset": dataset,
                "temperature": temperature,
                "num_runs": n_runs,
                "num_samples": n_samples,
                "overall_accuracy": round(overall_acc, 6),
                "pass@1": round(pass1, 6),
                "pass@3": round(pass3, 6) if not np.isnan(pass3) else None,
                "pass@5": round(pass5, 6) if not np.isnan(pass5) else None,
                "pass@10": round(pass10, 6) if not np.isnan(pass10) else None,
            })

        except Exception as e:
            print(f"⚠️ 跳过 {dirpath}: {e}")
            continue

    # 保存
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\n✅ 成功汇总 {len(df)} 个实验结果到 {output_csv}")
    else:
        print("❌ 未找到任何有效实验目录")

if __name__ == "__main__":
    main()