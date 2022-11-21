#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1

USER=$1
export MODEL_NAME=allenai/led-large-16384-arxiv
#export MODEL_NAME=allenai/led-base-16384
export DS_DIR=/disk1/sajad/datasets/sci/mup/hf_format/
export HF_DATASETS_CACHE=/disk0/$USER/.cache/huggingface2/
# 4190
# 8380

######### args

#export MAX_SRC=5120
#export VAL_STEPS=9530

export MAX_SRC=4096
export VAL_STEPS=9530
#export VAL_STEPS=10


#SP:400f24337c27f8f1fbb40ba7dd6c2a7c92b7a32f
#python -m torch.distributed.launch --nproc_per_node=2 run_summarization.py \
CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --ext_tokenizer_name roberta-large \
    --output_dir /disk0/$USER/.cache/sci-trained-models/mup-extdec-$MAX_SRC-1024 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --adam_beta2 0.999 \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 1 \
    --save_total_limit 5 \
    --text_column source \
    --summary_column summary \
    --overwrite_output_dir \
    --evaluation_strategy steps --warmup_ratio 0.05 --logging_steps 100 \
    --predict_with_generate \
    --max_grad_norm 1 \
    --lr_scheduler_type linear \
    --eval_steps $VAL_STEPS --save_steps $VAL_STEPS \
    --train_file $DS_DIR/train-sumSent.parquet \
    --validation_file $DS_DIR/val-sumSent.parquet \
    --do_eval \
    --max_source_length $MAX_SRC \
    --preprocessing_num_workers 4 \
    --metric_for_best_model rouge1_f \
    --greater_is_better True \
    --labeling dynamic \
    --do_train \
    --report_to wandb \
    --run_name mup-extdec-$MAX_SRC-1024 \

    #    --test_file $DS_DIR/test.reduced.complete.parquet \
#    --do_predict \
#    --filtered_ids "7cbbcd36c5af118c7caad20f1b2cf159"
#    --filtered_ids "183a64018088087429e503d3f533ea89"

