
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT="pegasus-eval"
export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb

DT=$(date +%F_%H-%M)
RUNID=pegasus-xsum-${DT}
OUTDIR=/data1/vchua/jpq-nlp/pegasus-eval/${RUNID}
mkdir -p $OUTDIR

# nohup python run_summarization.py \
#     --model_name_or_path google/pegasus-xsum \
#     --dataset_name xsum \
#     --max_source_length 512 \
#     --max_target_length 64 \
#     --do_eval \
#     --per_device_eval_batch_size 16 \
#     --predict_with_generate \
#     --num_beams 8 \
#     --overwrite_output_dir \
#     --run_name $RUNID \
#     --output_dir $OUTDIR > $OUTDIR/run.log 2>&1 &


nohup python run_summarization.py \
    --model_name_or_path google/pegasus-xsum \
    --dataset_name xsum \
    --max_source_length 512 \
    --max_target_length 56 \
    --do_eval \
    --per_device_eval_batch_size 16 \
    --predict_with_generate \
    --num_beams 8 \
    --overwrite_output_dir \
    --run_name $RUNID \
    --output_dir $OUTDIR > $OUTDIR/run.log 2>&1 &