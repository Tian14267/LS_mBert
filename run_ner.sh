#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
Output_dir="./outputs/panx_paper"
python ner_paper.py \
    --data_dir "./download/panx_udpipe_processed" \
    --model_name_or_path "./bert-base-multilingual-cased" \
    --per_gpu_train_batch_size 64 \
    --covert_rate 0.5 \
    --do_train \
    --do_eval \
    --do_predict \
    --train_langs "en" \
    --output_dir $Output_dir \
    --num_train_epochs 10 \
    2>&1 | tee ./$Output_dir/output.log;

