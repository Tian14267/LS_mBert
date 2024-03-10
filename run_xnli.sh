#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$HOME_DIR;
Output_dir="./outputs/xnli_paper"
python classify_xnli_paper.py \
    --data_dir "./download/xnli_udpipe_processed" \
    --model_name_or_path "/data/fffan/0_DeepLearning_code/7_NLP相关/6_cross_lingual_NLP/bert-base-multilingual-cased" \
    --num_train_epochs 10 \
    --covert_rate 0.5 \
    --train_language "en" \
    --per_gpu_train_batch_size 64 \
    --output_dir $Output_dir \
    2>&1 | tee $Output_dir/output.log;

