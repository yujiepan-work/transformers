#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778

# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="w2v2opt-ks ($(hostname))"
export CUDA_VISIBLE_DEVICES=0

NEPOCH=30
RUNID=run10.2-w2v2b-ks-qat-${NEPOCH}eph-baseline
NNCFCFG=/data/vchua/dev/optimum-ov/optimum-openvino/optimum/intel/nncf/configs/nncf_wav2vec2_config.json

# RUNID=run12b-w2v2b-ks-qat-${NEPOCH}eph-symm-act-signed-per-chl-w-ignoreFE
# RUNID=run23-w2v2b-ks-qat-${NEPOCH}eph-symm-act-signed-per-chl-w-ignoreFE-customkd-bt-tratio0.9
# NNCFCFG=/data/vchua/dev/optimum-ov/transformers/examples/pytorch/audio-classification/cfg-nncf/nncf_w2v2b_ks_qat.json

OUTROOT=/data/vchua/run/optimum-ov/w2v2b-ks
WORKDIR=/data/vchua/dev/optimum-ov/transformers/examples/pytorch/audio-classification

CONDAROOT=/data/vchua/miniconda3
CONDAENV=optimum-ov
# ---------------------------------------------------------------------------------------------

OUTDIR=$OUTROOT/$RUNID

# override label if in dryrun mode
if [[ $1 == "dryrun" ]]; then
    OUTDIR=$OUTROOT/dryrun-${RUNID}
    RUNID=dryrun-${RUNID}
fi

mkdir -p $OUTDIR
cd $WORKDIR
    # --distill_temp 5 \
    # --lr_scheduler_type cosine_with_restarts \
    # --teacher anton-l/wav2vec2-base-ft-keyword-spotting \
    # --teacher_ratio 0.9 \

cmd="
python run_audio_classification.py \
    --model_name_or_path anton-l/wav2vec2-base-ft-keyword-spotting \
    --dataset_name superb \
    --dataset_config_name ks \
    --remove_unused_columns False \
    --do_eval \
    --do_train \
    --nncf_config $NNCFCFG \
    --learning_rate 3e-5 \
    --max_length_seconds 1 \
    --attention_mask False \
    --warmup_ratio 0.1 \
    --num_train_epochs $NEPOCH \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 4 \
    --logging_strategy steps \
    --logging_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 200 \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --seed 0 \
    --run_name $RUNID \
    --output_dir $OUTDIR \
    --overwrite_output_dir
"
    # --save_total_limit 3 \

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