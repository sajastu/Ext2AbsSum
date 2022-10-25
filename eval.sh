#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1

MODEL_NAME=$1
#export MODEL_NAME=allenai/led-base-16384
DS_DIR=$2
TOPIC_FILE_PATH=$3

export HF_DATASETS_CACHE=/disk0/$USER/.cache/huggingface

CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --output_dir $MODEL_NAME \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --adam_beta2 0.999 \
    --text_column source \
    --summary_column summary \
    --overwrite_output_dir \
    --evaluation_strategy steps --warmup_ratio 0.05 --logging_steps 100 \
    --predict_with_generate \
    --max_grad_norm 1 \
    --lr_scheduler_type linear \
    --do_eval \
    --validation_file $DS_DIR/val.parquet \
    --max_source_length 6144 \
    --preprocessing_num_workers 4 \
    --mode test \
    --topic_file_path $TOPIC_FILE_PATH \

