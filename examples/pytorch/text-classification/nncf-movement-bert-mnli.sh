
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT="bert-mnli-mvmt"
export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb

NEPOCH=5
RUNID=bert-mnli-mvmt-${NEPOCH}eph-run5
OUTDIR=/data1/vchua/jpq-nlp/phase2/${RUNID}
mkdir -p $OUTDIR

nohup python run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name mnli \
    --do_eval \
    --do_train \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --learning_rate 3e-5 \
    --num_train_epochs $NEPOCH \
    --per_device_eval_batch_size 128 \
    --per_device_train_batch_size 64 \
    --max_seq_length 128 \
    --save_steps 3000 \
    --nncf_config mvmt_cfg/nncf_bert_xnli_mvmt.json \
    --logging_steps 1 \
    --overwrite_output_dir \
    --run_name $RUNID \
    --output_dir $OUTDIR 2>&1 | tee $OUTDIR/run.log &

    # --lr_scheduler_type cosine_with_restarts \
    # --warmup_ratio 0.05 \