#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778
export WANDB_PROJECT="tld-poc-post-training"

export CUDA_VISIBLE_DEVICES=3
WORKDIR=/home/vchua/tld-poc/transformers/examples/pytorch/question-answering

#----------------------------------------------------------------
cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/0%_Global_Normed_Abs_Ranking_FP32.json" \
    --overwrite_output_dir \
    --run_name "0%_Global_Normed_Abs_Ranking_FP32" \
    --output_dir "/data1/vchua/tld-poc-post-training/0%_Global_Normed_Abs_Ranking_FP32"

cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/0%_Global_Normed_Abs_Ranking_8bit.json" \
    --overwrite_output_dir \
    --run_name "0%_Global_Normed_Abs_Ranking_8bit" \
    --output_dir "/data1/vchua/tld-poc-post-training/0%_Global_Normed_Abs_Ranking_8bit"

#----------------------------------------------------------------
cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/10%_Global_Normed_Abs_Ranking_FP32.json" \
    --overwrite_output_dir \
    --run_name "10%_Global_Normed_Abs_Ranking_FP32" \
    --output_dir "/data1/vchua/tld-poc-post-training/10%_Global_Normed_Abs_Ranking_FP32"

cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/10%_Global_Normed_Abs_Ranking_8bit.json" \
    --overwrite_output_dir \
    --run_name "10%_Global_Normed_Abs_Ranking_8bit" \
    --output_dir "/data1/vchua/tld-poc-post-training/10%_Global_Normed_Abs_Ranking_8bit"

#----------------------------------------------------------------
cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/20%_Global_Normed_Abs_Ranking_FP32.json" \
    --overwrite_output_dir \
    --run_name "20%_Global_Normed_Abs_Ranking_FP32" \
    --output_dir "/data1/vchua/tld-poc-post-training/20%_Global_Normed_Abs_Ranking_FP32"

cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/20%_Global_Normed_Abs_Ranking_8bit.json" \
    --overwrite_output_dir \
    --run_name "20%_Global_Normed_Abs_Ranking_8bit" \
    --output_dir "/data1/vchua/tld-poc-post-training/20%_Global_Normed_Abs_Ranking_8bit"

#----------------------------------------------------------------
cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/30%_Global_Normed_Abs_Ranking_FP32.json" \
    --overwrite_output_dir \
    --run_name "30%_Global_Normed_Abs_Ranking_FP32" \
    --output_dir "/data1/vchua/tld-poc-post-training/30%_Global_Normed_Abs_Ranking_FP32"

cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/30%_Global_Normed_Abs_Ranking_8bit.json" \
    --overwrite_output_dir \
    --run_name "30%_Global_Normed_Abs_Ranking_8bit" \
    --output_dir "/data1/vchua/tld-poc-post-training/30%_Global_Normed_Abs_Ranking_8bit"

#----------------------------------------------------------------
cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/40%_Global_Normed_Abs_Ranking_FP32.json" \
    --overwrite_output_dir \
    --run_name "40%_Global_Normed_Abs_Ranking_FP32" \
    --output_dir "/data1/vchua/tld-poc-post-training/40%_Global_Normed_Abs_Ranking_FP32"

cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/40%_Global_Normed_Abs_Ranking_8bit.json" \
    --overwrite_output_dir \
    --run_name "40%_Global_Normed_Abs_Ranking_8bit" \
    --output_dir "/data1/vchua/tld-poc-post-training/40%_Global_Normed_Abs_Ranking_8bit"

#----------------------------------------------------------------
cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/50%_Global_Normed_Abs_Ranking_FP32.json" \
    --overwrite_output_dir \
    --run_name "50%_Global_Normed_Abs_Ranking_FP32" \
    --output_dir "/data1/vchua/tld-poc-post-training/50%_Global_Normed_Abs_Ranking_FP32"

cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/50%_Global_Normed_Abs_Ranking_8bit.json" \
    --overwrite_output_dir \
    --run_name "50%_Global_Normed_Abs_Ranking_8bit" \
    --output_dir "/data1/vchua/tld-poc-post-training/50%_Global_Normed_Abs_Ranking_8bit"

#----------------------------------------------------------------
cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/60%_Global_Normed_Abs_Ranking_FP32.json" \
    --overwrite_output_dir \
    --run_name "60%_Global_Normed_Abs_Ranking_FP32" \
    --output_dir "/data1/vchua/tld-poc-post-training/60%_Global_Normed_Abs_Ranking_FP32"

cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/60%_Global_Normed_Abs_Ranking_8bit.json" \
    --overwrite_output_dir \
    --run_name "60%_Global_Normed_Abs_Ranking_8bit" \
    --output_dir "/data1/vchua/tld-poc-post-training/60%_Global_Normed_Abs_Ranking_8bit"

#----------------------------------------------------------------
cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/70%_Global_Normed_Abs_Ranking_FP32.json" \
    --overwrite_output_dir \
    --run_name "70%_Global_Normed_Abs_Ranking_FP32" \
    --output_dir "/data1/vchua/tld-poc-post-training/70%_Global_Normed_Abs_Ranking_FP32"

cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/70%_Global_Normed_Abs_Ranking_8bit.json" \
    --overwrite_output_dir \
    --run_name "70%_Global_Normed_Abs_Ranking_8bit" \
    --output_dir "/data1/vchua/tld-poc-post-training/70%_Global_Normed_Abs_Ranking_8bit"

#----------------------------------------------------------------
cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/80%_Global_Normed_Abs_Ranking_FP32.json" \
    --overwrite_output_dir \
    --run_name "80%_Global_Normed_Abs_Ranking_FP32" \
    --output_dir "/data1/vchua/tld-poc-post-training/80%_Global_Normed_Abs_Ranking_FP32"

cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/80%_Global_Normed_Abs_Ranking_8bit.json" \
    --overwrite_output_dir \
    --run_name "80%_Global_Normed_Abs_Ranking_8bit" \
    --output_dir "/data1/vchua/tld-poc-post-training/80%_Global_Normed_Abs_Ranking_8bit"

#----------------------------------------------------------------
cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/90%_Global_Normed_Abs_Ranking_FP32.json" \
    --overwrite_output_dir \
    --run_name "90%_Global_Normed_Abs_Ranking_FP32" \
    --output_dir "/data1/vchua/tld-poc-post-training/90%_Global_Normed_Abs_Ranking_FP32"

cd $WORKDIR
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config "nncf_post_training/90%_Global_Normed_Abs_Ranking_8bit.json" \
    --overwrite_output_dir \
    --run_name "90%_Global_Normed_Abs_Ranking_8bit" \
    --output_dir "/data1/vchua/tld-poc-post-training/90%_Global_Normed_Abs_Ranking_8bit"