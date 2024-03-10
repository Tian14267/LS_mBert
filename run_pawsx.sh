export PYTHONPATH=$HOME_DIR;

export CUDA_VISIBLE_DEVICES=0
Output_dir="./outputs/pawsx_paper"
python classify_pawsx_paper.py \
    --data_dir "./download/pawsx_udpipe_processed" \
    --model_name_or_path "/data/fffan/0_DeepLearning_code/7_NLP相关/6_cross_lingual_NLP/bert-base-multilingual-cased" \
    --train_language "en" \
    --num_train_epochs 10 \
    --covert_rate 0.5 \
    --per_gpu_train_batch_size 96 \
    --output_dir $Output_dir\
    2>&1 | tee $SAVE_DIR/output.log;





