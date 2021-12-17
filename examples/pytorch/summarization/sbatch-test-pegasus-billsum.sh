#!/usr/bin/env bash

HOME=/data1/vchua
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate dgx4-pegasus-hf4p13

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=$(hostname)_pegasus_test
export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=711a0d50fe1ca61fd3a1be54c82ef54dd9432aff

DT=$(date +%F_%H-%M)
RUNID=pegasus-billsum-${DT}
OUTDIR=/data1/vchua/runs-pegasus-hf4p13/pegasus-test/${RUNID}
WORKDIR=/data1/vchua/dgx4-pegasus-hf4p13/transformers/examples/pytorch/summarization
CKPT=google/pegasus-billsum

mkdir -p $OUTDIR
cd $WORKDIR

python run_summarization.py \
    --model_name_or_path $CKPT \
    --dataset_name billsum \
    --max_source_length 1024 \
    --max_target_length 256 \
    --do_predict \
    --per_device_eval_batch_size 8 \
    --predict_with_generate \
    --num_beams 8 \
    --overwrite_output_dir \
    --run_name $RUNID \
    --output_dir $OUTDIR > $OUTDIR/run.log 2>&1 

