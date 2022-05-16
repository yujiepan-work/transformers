#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778

# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="(hgx1) nncf-hs"
export CUDA_VISIBLE_DEVICES=0

NEPOCH=2
TBS=16
EBS=64
SL=384
DS=128

# TEACHER=vuiseng9/roberta-l-squadv1.1
# TEMPERATURE=5
# ALPHA=0.9

BASEM=bert-large-uncased-whole-word-masking
# BASEM=bert-base-uncased
# BASEM=bert-large-uncased-whole-word-masking-finetuned-squad
RUNID=run02-bert-l-pt-squadv1.1-nncf-hs-sl${SL}-ds${DS}-e${NEPOCH}-tbs${TBS}-refactor-stats2
NNCFCFG=/data/vchua/dev/jpqd-alpha/transformers/examples/pytorch/question-answering/rb-bert-squad.json

OUTROOT=/data/vchua/run/jpqd-alpha/nncf-hs
WORKDIR=/data/vchua/dev/jpqd-alpha/transformers/examples/pytorch/question-answering

CONDAROOT=/data/vchua/miniconda3
CONDAENV=jpqd-alpha
# ---------------------------------------------------------------------------------------------

OUTDIR=$OUTROOT/$RUNID

# override label if in dryrun mode
if [[ $1 == "dryrun" ]]; then
    OUTDIR=$OUTROOT/dryrun-${RUNID}
    RUNID=dryrun-${RUNID}
fi

mkdir -p $OUTDIR
cd $WORKDIR

    # --teacher $TEACHER \
    # --teacher_ratio $ALPHA \
    # --distill_temp $TEMPERATURE \
cmd="
python run_qa.py \
    --model_name_or_path ${BASEM} \
    --dataset_name squad \
    --do_eval \
    --do_train \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --learning_rate 3e-5 \
    --fp16 \
    --num_train_epochs $NEPOCH \
    --per_device_eval_batch_size $EBS \
    --per_device_train_batch_size $TBS \
    --max_seq_length $SL \
    --doc_stride $DS \
    --save_steps 1000 \
    --logging_steps 1 \
    --overwrite_output_dir \
    --nncf_config $NNCFCFG \
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
