
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT="pegasus"
export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb

NEPOCH=5
RUNID=pegasus-xsum-${NEPOCH}eph-run1
OUTDIR=/data1/vchua/jpq-nlp/pegasus/${RUNID}
mkdir -p $OUTDIR

nohup python run_summarization.py \
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
    --evaluation_strategy epoch \
    --save_strategy steps \
    --save_steps 10000 \
    --logging_steps 1 \
    --overwrite_output_dir \
    --run_name $RUNID \
    --output_dir $OUTDIR > $OUTDIR/run.log 2>&1 &
#2>&1 | tee $OUTDIR/run.log &

    # --lr_scheduler_type cosine_with_restarts \
    # --warmup_ratio 0.05 \
