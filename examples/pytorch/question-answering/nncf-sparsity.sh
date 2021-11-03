#!/usr/bin/env bash

NEPOCH=5
OUTDIR=/data1/vchua/jpq-nlp/bert-base-rb-0.5sparse-${NEPOCH}eph
mkdir -p $OUTDIR

python run_qa.py \
    --model_name_or_path vuiseng9/bert-base-uncased-squad \
    --dataset_name squad \
    --do_eval \
    --do_train \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --learning_rate 3e-5 \
    --num_train_epochs $NEPOCH \
    --per_device_eval_batch_size 128 \
    --per_device_train_batch_size 6 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --save_steps 2500 \
    --logging_steps 1 \
    --nncf_config ../../../nncf_bert_config_squad_sparsity.json \
    --overwrite_output_dir \
    --output_dir $OUTDIR 2>&1 | tee $OUTDIR/run.log &
