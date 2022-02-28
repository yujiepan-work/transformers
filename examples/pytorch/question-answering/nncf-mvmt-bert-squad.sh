#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778

# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="nncf-mvmt-squad ($(hostname))"
export CUDA_VISIBLE_DEVICES=0

# NEPOCH=20
# RUNID=run27.sat-bert-squad-nncf-mvmt-lt-${NEPOCH}eph-r0.02-threshold-end-3eph-prune-bias-filled
# NNCFCFG=mvmt_cfg/ref_bert_squad_nncf_mvmt.json

# NEPOCH=4
# RUNID=run32.fri-bert-squad-nncf-mvmt-lt-${NEPOCH}eph-r0.005-threshold-end-3eph-functest-prune-bias-filled
# NNCFCFG=mvmt_cfg/functest-bert_squad_nncf_mvmt.json

NEPOCH=20
RUNID=run27.sun-bert-squad-nncf-mvmt-8bit-lt-${NEPOCH}eph-r0.02-threshold-end-10eph-prune-bias-prefilled
NNCFCFG=mvmt_cfg/8bit_ref_bert_squad_nncf_mvmt.json

OUTROOT=/data2/vchua/run/ssbs-feb/bert-squad
WORKDIR=/data2/vchua/dev/ssbs-feb/transformers/examples/pytorch/question-answering

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
python run_qa.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name squad \
    --do_eval \
    --do_train \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --learning_rate 3e-5 \
    --teacher bert-large-uncased-whole-word-masking-finetuned-squad \
    --teacher_ratio 0.9 \
    --num_train_epochs $NEPOCH \
    --per_device_eval_batch_size 128 \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config $NNCFCFG \
    --save_steps 2500 \
    --logging_steps 1 \
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

# --teacher vuiseng9/bert-base-uncased-squad \
# --teacher_ratio 0.9 \
# --lr_scheduler_type cosine_with_restarts \
# --warmup_ratio 0.05 \

    # --optimize_model_before_eval  \
    # --optimized_checkpoint /data2/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled  \