#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778

# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="nncf-mvmt-mnli ($(hostname))"
export CUDA_VISIBLE_DEVICES=0

NEPOCH=12

# RUNID=run00-bert-mnli-mvmt-customkd-bt-${NEPOCH}eph
RUNID=run65-bert-mnli-nncf-mvmt-bt-${NEPOCH}eph-r0.005-threshold-end-5eph-exclude-pooler-regression
OUTROOT=/data1/vchua/run/ssbs-feb/bert-mnli
WORKDIR=/data1/vchua/dev/ssbs-feb/transformers/examples/pytorch/text-classification

CONDAROOT=/data1/vchua/miniconda3
CONDAENV=ssbs-feb
# ---------------------------------------------------------------------------------------------

OUTDIR=$OUTROOT/$RUNID

# override label if in dryrun mode
if [[ $1 == "dryrun" ]]; then
    OUTDIR=$OUTROOT/dryrun-${RUNID}
    RUNID=dryrun-${RUNID}
fi

mkdir -p $OUTDIR
cd $WORKDIR

cmd="
python run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name mnli \
    --max_seq_length 128 \
    --do_train \
    --per_device_eval_batch_size 128 \
    --learning_rate 3e-5 \
    --num_train_epochs $NEPOCH \
    --logging_steps 1 \
    --teacher vuiseng9/bert-mnli \
    --teacher_ratio 0.9 \
    --do_eval \
    --per_device_train_batch_size 32 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --nncf_config mvmt_cfg/bert_mnli_nncf_mvmt.json \
    --save_steps 2500 \
    --overwrite_output_dir \
    --run_name $RUNID \
    --output_dir $OUTDIR
"

if [[ $1 == "local" ]]; then
    echo "${cmd}" > $OUTDIR/run.log
    echo "### End of CMD ---" >> $OUTDIR/run.log
    cmd="nohup ${cmd}"
    eval $cmd >> $OUTDIR/run.log 2>&1 &
    echo "logpath: $OUTDIR/run.log"
elif [[ $1 == "dryrun" ]]; then
    echo "[INFO: dryrun, add --max_steps 25 to cli"
    cmd="${cmd} --max_steps 25"
    echo "${cmd}" > $OUTDIR/dryrun.log
    echo "### End of CMD ---" >> $OUTDIR/dryrun.log
    eval $cmd >> $OUTDIR/dryrun.log 2>&1 &
    echo "logpath: $OUTDIR/dryrun.log"
else
    source $CONDAROOT/etc/profile.d/conda.sh
    conda activate ${CONDAENV}
    eval $cmd
fi

    # --lr_scheduler_type cosine_with_restarts \
    # --warmup_ratio 0.05 \