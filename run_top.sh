#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
Output_dir="./outputs/mtop_paper"
python mtop_paper.py \
    --data_dir "./download/mtop_udpipe_processed_en" \
    --model_name_or_path "/data/fffan/0_DeepLearning_code/7_NLP相关/6_cross_lingual_NLP/bert-base-multilingual-cased" \
    --intent_labels "./download/mtop_udpipe_processed_en/intent_label.txt" \
    --slot_labels "./download/mtop_udpipe_processed_en/slot_label.txt" \
    --per_gpu_train_batch_size 64 \
    --do_train \
    --do_eval \
    --do_predict \
    --train_langs "en" \
    --output_dir $Output_dir \
    --num_train_epochs 10 \
    2>&1 | tee ./$Output_dir/output.log;



