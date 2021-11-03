
#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=3
# NEPOCH=10
# OUTDIR=/data1/vchua/jpq-nlp/bert-base-mvmt-${NEPOCH}eph-run5
# mkdir -p $OUTDIR

# python run_qa.py \
#     --model_name_or_path bert-base-uncased \
#     --dataset_name squad \
#     --do_eval \
#     --do_train \
#     --evaluation_strategy steps \
#     --eval_steps 250 \
#     --learning_rate 3e-5 \
#     --num_train_epochs $NEPOCH \
#     --per_device_eval_batch_size 128 \
#     --per_device_train_batch_size 16 \
#     --max_seq_length 384 \
#     --doc_stride 128 \
#     --save_steps 2500 \
#     --nncf_config ../../../nncf_bert_config_squad_mvnt_pruning.json \
#     --logging_steps 1 \
#     --overwrite_output_dir \
#     --output_dir $OUTDIR 2>&1 | tee $OUTDIR/run.log &

export CUDA_VISIBLE_DEVICES=3
NEPOCH=10
OUTDIR=/data1/vchua/jpq-nlp/bert-base-mvmt-${NEPOCH}eph-distillation-run1
mkdir -p $OUTDIR

python run_qa.py \
    --model_name_or_path bert-base-uncased \
    --teacher vuiseng9/bert-base-uncased-squad \
    --teacher_ratio 0.9 \
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
    --nncf_config ../../../nncf_bert_config_squad_mvnt_pruning.json \
    --logging_steps 1 \
    --overwrite_output_dir \
    --output_dir $OUTDIR 2>&1 | tee $OUTDIR/run.log &