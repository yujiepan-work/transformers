
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=$(hostname)_pegasus_ft
export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb

NEPOCH=5
RUNID=pegasus-arxiv-${NEPOCH}eph-run1
OUTDIR=/data1/vchua/pegasus-hf4p13/pegasus/${RUNID}
mkdir -p $OUTDIR

nohup python run_summarization.py \
    --model_name_or_path google/pegasus-large \
    --dataset_name ccdv/arxiv-summarization \
    --do_train \
    --adafactor \
    --learning_rate 8e-4 \
    --label_smoothing_factor 0.1 \
    --num_train_epochs $NEPOCH \
    --per_device_train_batch_size 2 \
    --do_eval \
    --per_device_eval_batch_size 2 \
    --num_beams 8 \
    --max_source_length 1024 \
    --max_target_length 256 \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --save_strategy steps \
    --save_steps 5000 \
    --logging_steps 1 \
    --overwrite_output_dir \
    --run_name $RUNID \
    --output_dir $OUTDIR > $OUTDIR/run.log 2>&1 &