#!/bin/bash

# 设置 Python 解释器路径
PYTHON="/home/xsj/data_xsj/miniconda3/envs/fast/bin/python"

# 模型列表：
# - 第一个是本地路径
# - 后两个是 Hugging Face 模型 ID（vLLM 会自动下载）
models=(
  "/home/xsj/data_xsj/1models/Qwen3-0.6B"
  "/home/xsj/data_xsj/1models/Qwen3-1.7B"
  # "/home/xsj/data_xsj/1models/DeepSeek-R1-Distill-Qwen-1.5B"
)

# 数据集列表
datasets=(
  "opencompass/AIME2025"
  "HuggingFaceH4/aime_2024"
  "FlagEval/HMMT_2025"
  # "HuggingFaceH4/MATH-500"
)

# 温度列表
temperatures=(0.4 0.6 0.8 1.0)

# 其他固定参数
NUM_SAMPLES=10
OUTPUT_BASE_DIR="results90"

# 确保输出目录存在
mkdir -p "$OUTPUT_BASE_DIR"

# 嵌套循环：模型 × 数据集 × 温度
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for temp in "${temperatures[@]}"; do
      echo "🚀 Running: model=${model}, dataset=${dataset}, temperature=${temp}"

      CUDA_VISIBLE_DEVICES=1,2 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      "$PYTHON" attention_temperature3.py \
        --model_name "$model" \
        --dataset "$dataset" \
        --do_sample \
        --num_samples "$NUM_SAMPLES" \
        --temperature "$temp" \
        --output_base_dir "$OUTPUT_BASE_DIR"

      echo "✅ Finished: ${model##*/} | ${dataset##*/} | T=${temp}"
      echo "----------------------------------------"
    done
  done
done

echo "🎉 All experiments completed!"