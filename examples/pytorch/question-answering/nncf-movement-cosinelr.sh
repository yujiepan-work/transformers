#!/usr/bin/env bash

HOME=/data1/vchua
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate jpq-nlp

export CUDA_VISIBLE_DEVICES=0
NEPOCH=9
OUTDIR=/data1/vchua/runs-coslr-jpq-nlp/bert-base-mvmt-${NEPOCH}eph-PruneByBlock-run1
mkdir -p $OUTDIR

cd /data1/vchua/jpq-nlp/transformers/examples/pytorch/question-answering
python run_qa.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name squad \
    --do_eval \
    --do_train \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --learning_rate 3e-5 \
    --lr_scheduler_type cosine_with_restarts \
    --warmup_ratio 0.05 \
    --num_train_epochs $NEPOCH \
    --per_device_eval_batch_size 128 \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --save_steps 2500 \
    --nncf_config ../../../nncf_bert_config_squad_mvnt_pruning.json \
    --logging_steps 1 \
    --overwrite_output_dir \
    --output_dir $OUTDIR | tee $OUTDIR/run.log

# 2>&1 &