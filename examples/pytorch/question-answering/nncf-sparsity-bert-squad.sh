
#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778

# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="tld-poc ($(hostname))"
export CUDA_VISIBLE_DEVICES=3
NEPOCH=10
SPARSITY=50pc
RUNID=run01-bert-squad-cropped-magsparse-${SPARSITY}-${NEPOCH}eph
OUTROOT=/data1/vchua/tld-poc-$(hostname)
WORKDIR=/home/vchua/tld-poc/transformers/examples/pytorch/question-answering

CONDAROOT=/data1/vchua
CONDAENV=tld-poc
# ---------------------------------------------------------------------------------------------
OUTDIR=$OUTROOT/$RUNID
mkdir -p $OUTDIR

cd $WORKDIR

cmd="
python run_qa.py \
    --model_name_or_path /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-compiled \
    --optimize_model_before_eval \
    --optimized_checkpoint /data1/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled \
    --dataset_name squad \
    --do_eval \
    --do_train \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --learning_rate 3e-5 \
    --num_train_epochs $NEPOCH \
    --per_device_eval_batch_size 128 \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --save_steps 2500 \
    --nncf_config nncf_bert_squad_sparsity.json \
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
else
    source $CONDAROOT/miniconda3/etc/profile.d/conda.sh
    conda activate ${CONDAENV}
    eval $cmd
fi

# --teacher vuiseng9/bert-base-uncased-squad \
# --teacher_ratio 0.9 \
# --lr_scheduler_type cosine_with_restarts \
# --warmup_ratio 0.05 \
