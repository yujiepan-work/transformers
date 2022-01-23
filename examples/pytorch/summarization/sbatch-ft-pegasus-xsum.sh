#!/usr/bin/env bash

HOME=/data1/vchua
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate dgx4-pegasus-hf4p13

WORKDIR=/data1/vchua/dgx4-pegasus-hf4p13/transformers/examples/pytorch/summarization

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=$(hostname)_pegasus_ft
export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778

NEPOCH=10
RUNID=pegasus-xsum-${NEPOCH}eph-run1
OUTDIR=/data1/vchua/runs-pegasus-hf4p13/pegasus-ft/${RUNID}
mkdir -p $OUTDIR

cd $WORKDIR

python run_summarization.py \
    --model_name_or_path google/pegasus-large \
    --dataset_name xsum \
    --do_train \
    --adafactor \
    --learning_rate 1e-4 \
    --label_smoothing_factor 0.1 \
    --num_train_epochs $NEPOCH \
    --per_device_train_batch_size 8 \
    --do_eval \
    --per_device_eval_batch_size 8 \
    --num_beams 8 \
    --max_source_length 512 \
    --max_target_length 64 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_strategy steps \
    --save_steps 2000 \
    --logging_steps 1 \
    --overwrite_output_dir \
    --run_name $RUNID \
    --output_dir $OUTDIR > $OUTDIR/run.log 2>&1
