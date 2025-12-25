PYTHON="/root/miniconda3/bin/python"

models=(
  # "/home/xsj/data_xsj/1models/Qwen3-0.6B"
  # "/root/autodl-tmp/Qwen3-1.7B"
  "/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-1.5B"
  # "/root/autodl-tmp/Qwen2.5-Math-1.5B"
  # "/home/xsj/data_xsj/1models/DeepSeek-R1-Distill-Qwen-1.5B"
)

# 数据集列表
datasets=(
  # "opencompass/AIME2025"
  # "HuggingFaceH4/aime_2024"
  # "FlagEval/HMMT_2025"
  "HuggingFaceH4/MATH-500"
)

# 温度列表
temperatures=(0.4 0.6)  

# 其他固定参数
NUM_SAMPLES=10
OUTPUT_BASE_DIR="results"

# 确保输出目录存在
mkdir -p "$OUTPUT_BASE_DIR"

# 嵌套循环：模型 × 数据集 × 温度
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for temp in "${temperatures[@]}"; do
      echo "🚀 Running: model=${model}, dataset=${dataset}, temperature=${temp}"
      "$PYTHON" attention_temperature2.py \
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