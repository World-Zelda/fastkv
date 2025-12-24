import pandas as pd
import numpy as np
from typing import List, Union
import itertools
from math import comb
import os

def pass_at_k(results: Union[List[List[int]], np.ndarray], k: int = 1) -> float:
    """
    计算 pass@k 指标
    
    参数:
    results: n*m 矩阵，n次运行，m个样本
            每个元素: 0表示失败，1表示成功
    k: 考虑的尝试次数
    
    返回:
    pass@k 值
    """
    # 转换为numpy数组
    if isinstance(results, list):
        results = np.array(results)
    
    n, m = results.shape
    
    if k > n:
        raise ValueError(f"k={k} 不能大于运行次数 n={n}")
    
    # 转置矩阵，使得每行代表一个样本在不同运行中的结果
    # 形状变为 m*n: m个样本，每个样本有n次运行结果
    sample_results = results.T  # 形状: (m, n)
    
    total_pass = 0
    
    for i in range(m):
        sample_scores = sample_results[i]  # 单个样本的n次运行结果
        
        # 计算在k次尝试中至少成功一次的概率
        # 方法1: 直接统计
        successes = np.sum(sample_scores)
        failures = n - successes
        
        if k > n - failures:  # 如果要尝试的次数大于成功次数
            # 肯定能成功
            total_pass += 1
        else:
            # 计算在k次尝试中全部失败的概率
            # 使用组合数计算
            fail_combinations = comb(failures, k) if failures >= k else 0
            total_combinations = comb(n, k)
            
            if total_combinations > 0:
                prob_all_fail = fail_combinations / total_combinations
                prob_at_least_one_success = 1 - prob_all_fail
                total_pass += prob_at_least_one_success
            else:
                total_pass += 0
    
    # 计算平均pass@k
    pass_at_k_score = total_pass / m
    
    return pass_at_k_score

def pass_at_k_exact(results: Union[List[List[int]], np.ndarray], k: int = 1) -> float:
    """
    精确计算 pass@k 指标（通过模拟）
    
    参数:
    results: n*m 矩阵
    k: 考虑的尝试次数
    
    返回:
    pass@k 值
    """
    if isinstance(results, list):
        results = np.array(results)
    
    n, m = results.shape
    
    if k > n:
        raise ValueError(f"k={k} 不能大于运行次数 n={n}")
    
    # 转置矩阵
    sample_results = results.T  # 形状: (m, n)
    
    total_pass = 0
    
    for i in range(m):
        sample_scores = sample_scores[i]
        successes = np.sum(sample_scores)
        
        if successes == 0:
            # 没有成功，概率为0
            total_pass += 0
        elif successes >= k:
            # 如果成功次数大于等于k，肯定能成功
            total_pass += 1
        else:
            # 需要精确计算
            # 生成所有可能的k次尝试组合
            success_in_k = 0
            total_combinations = 0
            
            # 对于较小的n，可以直接枚举
            indices = list(range(n))
            for combo in itertools.combinations(indices, k):
                total_combinations += 1
                # 检查这个组合中是否有成功
                if any(sample_scores[idx] == 1 for idx in combo):
                    success_in_k += 1
            
            if total_combinations > 0:
                total_pass += success_in_k / total_combinations
    
    return total_pass / m

def pass_at_k_multiple_k(results: Union[List[List[int]], np.ndarray], k_values: List[int] = None) -> dict:
    """
    计算多个k值的pass@k
    
    参数:
    results: n*m 矩阵
    k_values: 要计算的k值列表
    
    返回:
    包含不同k值结果的字典
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]
    
    scores = {}
    for k in k_values:
        try:
            scores[f'pass@{k}'] = pass_at_k(results, k)
        except ValueError as e:
            print(f"跳过 k={k}: {e}")
            scores[f'pass@{k}'] = None
    
    return scores

def analyze_results(results: Union[List[List[int]], np.ndarray]):
    """
    分析结果矩阵的统计信息
    """
    if isinstance(results, list):
        results = np.array(results)
    
    n, m = results.shape
    
    print(f"结果矩阵形状: {n}次运行 × {m}个样本")
    print(f"总运行次数: {n * m}")
    print(f"总体成功率: {np.mean(results):.4f}")
    print(f"各次运行的平均成功率: {np.mean(results, axis=1)}")
    print(f"各样本的成功率: {np.mean(results, axis=0)}")
    
    # 计算不同k值的pass@k
    max_k = min(10, n)  # 最多计算到k=10或n
    k_vals = [k for k in [1, 3, 5, 10] if k <= n]
    
    scores = pass_at_k_multiple_k(results, k_vals)
    
    print("\npass@k 结果:")
    for k_str, score in scores.items():
        if score is not None:
            print(f"{k_str}: {score:.4f}")

def load_csvs_from_dir(dir_path: str) -> np.ndarray:
    """
    从指定目录加载所有 .csv 文件，按 global_index 对齐
    """
    csv_files = sorted([
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith('.csv')
    ])
    
    if len(csv_files) == 0:
        raise FileNotFoundError(f"在目录 {dir_path} 中未找到任何 .csv 文件")
    
    print(f"找到 {len(csv_files)} 个 CSV 文件：")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    
    all_data = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        if 'global_index' not in df.columns or 'score' not in df.columns:
            raise ValueError(f"文件 {file_path} 缺少 'global_index' 或 'score' 列")
        
        # 转为整数（0/1），处理 NaN（如 missing answer）
        df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0).astype(int)
        df = df[['global_index', 'score']].set_index('global_index')
        all_data.append(df)

    # 合并对齐
    all_indices = sorted(set(idx for df in all_data for idx in df.index))
    aligned = []
    for df in all_data:
        aligned_df = df.reindex(all_indices, fill_value=0)
        aligned.append(aligned_df['score'].values)
    
    return np.array(aligned)  # shape: (n_runs, n_samples)

if __name__ == "__main__":
    # 指定你的 CSV 所在目录
    csv_dir = "results/Qwen3-0.6B/MATH-500/1.0/0-500"
    
    # 加载数据
    results_matrix = load_csvs_from_dir(csv_dir)
    
    # 分析
    print("\n" + "="*50)
    analyze_results(results_matrix)
    print("="*50)

    # --- 新增：保存为 JSON ---
    import json

    n_runs, n_samples = results_matrix.shape
    overall_acc = float(np.mean(results_matrix))
    
    # 计算 pass@k（确保 k <= n_runs）
    k_vals = [k for k in [1, 3, 5, 10] if k <= n_runs]
    pass_at_k_scores = pass_at_k_multiple_k(results_matrix, k_vals)
    
    # 构建结果字典
    metrics = {
        "model": "Qwen3-0.6B",
        "dataset": "MATH-500",
        "temperature": 1.0,
        "num_runs": n_runs,
        "num_samples": n_samples,
        "overall_accuracy": round(overall_acc, 6),
        "pass_at_k": {
            k: round(v, 6) if v is not None else None
            for k, v in pass_at_k_scores.items()
        }
    }

    # 保存路径
    output_path = os.path.join(csv_dir, "metrics.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到: {output_path}")
