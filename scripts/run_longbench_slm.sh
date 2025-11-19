# This is example for LongBench script.

# Dataset List
# "narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"

# Model List
# nvidia/Llama-3.1-8B-UltraLong-1M-Instruct
# meta-llama/Meta-Llama-3.1-8B-Instruct
# meta-llama/Llama-3.2-3B-Instruct
# mistralai/Mistral-Nemo-Instruct-2407

# Config
dataset_list="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p" 
model="/home/xsj/data_xsj/1models/Phi-3.5-mini-instruct"
device=0
max_prompt=1024

# SnapKV
path="streamingllm-$max_prompt"
for dataset in $dataset_list
do
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.main \
    --model $model \
    --mode streamingllm \
    --pooling avgpool \
    --kernel_size 7 \
    --window_size 8 \
    --save_path $path \
    --dataset $dataset \
    --max_capacity_prompt $max_prompt
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.evaluate \
    --model $model \
    --eval_path $path
done

