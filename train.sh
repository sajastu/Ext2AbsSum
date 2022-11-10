#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1

USER=$1
export MODEL_NAME=allenai/led-large-16384-arxiv
#export MODEL_NAME=allenai/led-base-16384
export DS_DIR=/disk1/sajad/datasets/sci/mup/hf_format/
export HF_DATASETS_CACHE=/disk0/$USER/.cache/huggingface
# 4190
# 8380
#python -m torch.distributed.launch --nproc_per_node=2 run_summarization.py \
CUDA_VISIBLE_DEVICES=1 python run_summarization.py \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --ext_tokenizer_name roberta-large \
    --output_dir /disk0/$USER/.cache/sci-trained-models/test \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --adam_beta2 0.999 \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 8 \
    --save_total_limit 5 \
    --text_column source \
    --summary_column summary \
    --overwrite_output_dir \
    --evaluation_strategy steps --warmup_ratio 0.05 --logging_steps 10 \
    --predict_with_generate \
    --max_grad_norm 1 \
    --lr_scheduler_type linear \
    --eval_steps 200 --save_steps 4190 \
    --train_file $DS_DIR/val-sumSent.parquet \
    --validation_file $DS_DIR/val-sumSent.parquet \
    --do_eval \
    --max_source_length 3072 \
    --preprocessing_num_workers 1 \
    --metric_for_best_model rougeL_f \
    --greater_is_better True \
    --labeling dynamic \
    --do_train \
    #    --report_to wandb \
#    --run_name mup-led-arxiv-2048-6144-AllSents-PrepConc-fixed \

    #    --test_file $DS_DIR/test.reduced.complete.parquet \
#    --do_predict \
#    --filtered_ids "7cbbcd36c5af118c7caad20f1b2cf159"
#    --filtered_ids "183a64018088087429e503d3f533ea89"

