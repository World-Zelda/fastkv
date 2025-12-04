#!/bin/bash

# 设置环境变量（根据你的原始配置）
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 定义温度列表
temperatures=(0.4 0.6 0.8 1.0)
# temperatures=(0.4)

# 遍历每个温度值
for temp in "${temperatures[@]}"; do
    echo "🚀 Running with temperature = $temp"
    
    python attention_temperature2.py \
        --model_name "/root/autodl-fs/Qwen3-4B" \
        --do_sample \
        --num_samples 10 \
        --temperature "$temp" \
        --output_base_dir "results" \
        --start 0 \
        --end 500
    
    echo "✅ Finished temperature = $temp"
    echo "----------------------------------------"
done

echo "🎉 All temperature runs completed!"