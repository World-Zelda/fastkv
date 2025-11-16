dataset_list="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"

model="/root/autodl-tmp/Mistral-7B-Instruct-v0.3"
device=0

# 要测试的 max_prompt 值列表
max_prompt_list="129 256 512 1024"

# ===== SnapKV =====
for max_prompt in $max_prompt_list
do
    path="snapkv-$max_prompt"
    for dataset in $dataset_list
    do
        echo "Running SnapKV: dataset=$dataset, max_prompt=$max_prompt"
        CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.main \
            --model "$model" \
            --mode snapkv \
            --pooling avgpool \
            --kernel_size 7 \
            --window_size 8 \
            --save_path "$path/$dataset" \
            --dataset "$dataset" \
            --max_capacity_prompt "$max_prompt"

        CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.evaluate \
            --model "$model" \
            --eval_path "$path/$dataset"
    done
done

# ===== FullKV =====
path="fullkv"
for dataset in $dataset_list
do
    echo "Running FullKV: dataset=$dataset"
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.main \
        --model "$model" \
        --mode fullkv \
        --save_path "$path/$dataset" \
        --dataset "$dataset"

    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.evaluate \
        --model "$model" \
        --eval_path "$path/$dataset"
done