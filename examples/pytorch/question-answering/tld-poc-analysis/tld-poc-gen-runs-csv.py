
import pandas as pd
import numpy as np
import json
import wandb
import os

from collections import OrderedDict, Counter

api = wandb.Api()

print('-'*100)
# Post-training sparsity + quantization
#--------------------------------------
runs = api.runs("vchua/tld-poc-post-training")

pt_dict = OrderedDict() # post-training evaluation
for run_i, run in enumerate(runs):
    print(run_i, run.name)
    sparsity = float(run.name.split("%")[0])
    precision = run.name.split("_")[-1]

    task_metric_df = run.history(x_axis='train/global_step', keys=['eval/f1', 'eval/exact_match'])
    if len(task_metric_df) != 1:
        raise ValueError("This routine is specifically for run with single evaluation logged")
    
    pt_dict[run.name] = {   
        'sparsity': sparsity,
        'precision': precision,
        'f1': task_metric_df['eval/f1'][0],
        'em': task_metric_df['eval/exact_match'][0]
    }

pt_df = pd.DataFrame.from_dict(pt_dict, orient='index')

print('-'*100)
# nncf magnitude pruning + quantization + large teacher distillation
#--------------------------------------
runs = api.runs("vchua/tld-poc (csr-dgx1-04)")

sparse_8bit_nncfkd_dict = OrderedDict() # post-training evaluation

run_labels = ['run45', 'run46', 'run47', 'run48', 'run49']

for run_i, run in enumerate(runs):
    if run.name.split('-')[0] in run_labels:
        print(run_i, run.name)
        sparsity = float(run.name.split('-')[7].split('pc')[0])

        task_metric_df = run.history(x_axis='train/global_step', keys=['eval/f1', 'eval/exact_match'])

        global_step_with_max_f1 = np.argwhere((task_metric_df['eval/f1'] == task_metric_df['eval/f1'].max()).astype(float).to_numpy()).flatten().tolist()
        if len(global_step_with_max_f1) != 1:
            raise ValueError("There may be ties in maximum f1")
        
        entry = task_metric_df.loc[global_step_with_max_f1[0],]

        sparse_8bit_nncfkd_dict[run.name] = {   
            'sparsity': sparsity,
            'global_step': entry['train/global_step'],
            'f1': entry['eval/f1'],
            'em': entry['eval/exact_match']
        }

sparse_8bit_nncfkd_df = pd.DataFrame.from_dict(sparse_8bit_nncfkd_dict, orient='index')

print('-'*100)
# nncf magnitude pruning + quantization + base* teacher distillation
#--------------------------------------
runs = api.runs("vchua/tld-poc (csr-dgx1-04)")

sparse_8bit_bt_dict = OrderedDict() # post-training evaluation

run_labels = ['run25', 'run26', 'run27', 'run28', 'run29', 'run30']

for run_i, run in enumerate(runs):
    if run.name.split('-')[0] in run_labels:
        print(run_i, run.name)
        sparsity = float(run.name.split('-')[7].split('pc')[0])

        task_metric_df = run.history(x_axis='train/global_step', keys=['eval/f1', 'eval/exact_match'])

        global_step_with_max_f1 = np.argwhere((task_metric_df['eval/f1'] == task_metric_df['eval/f1'].max()).astype(float).to_numpy()).flatten().tolist()
        if len(global_step_with_max_f1) != 1:
            raise ValueError("There may be ties in maximum f1")
        
        entry = task_metric_df.loc[global_step_with_max_f1[0],]

        sparse_8bit_bt_dict[run.name] = {   
            'sparsity': sparsity,
            'global_step': entry['train/global_step'],
            'f1': entry['eval/f1'],
            'em': entry['eval/exact_match']
        }

sparse_8bit_bt_df = pd.DataFrame.from_dict(sparse_8bit_bt_dict, orient='index')

print('-'*100)
# nncf magnitude pruning + quantization + large* teacher distillation
#--------------------------------------
runs = api.runs("vchua/tld-poc (csr-dgx1-04)")

sparse_8bit_lt_dict = OrderedDict() # post-training evaluation

run_labels = ['run31', 'run32', 'run33', 'run34', 'run35', 'run36']

for run_i, run in enumerate(runs):
    if run.name.split('-')[0] in run_labels:
        print(run_i, run.name)
        sparsity = float(run.name.split('-')[8].split('pc')[0])

        task_metric_df = run.history(x_axis='train/global_step', keys=['eval/f1', 'eval/exact_match'])

        global_step_with_max_f1 = np.argwhere((task_metric_df['eval/f1'] == task_metric_df['eval/f1'].max()).astype(float).to_numpy()).flatten().tolist()
        if len(global_step_with_max_f1) != 1:
            raise ValueError("There may be ties in maximum f1")
        
        entry = task_metric_df.loc[global_step_with_max_f1[0],]

        sparse_8bit_lt_dict[run.name] = {   
            'sparsity': sparsity,
            'global_step': entry['train/global_step'],
            'f1': entry['eval/f1'],
            'em': entry['eval/exact_match']
        }

sparse_8bit_lt_df = pd.DataFrame.from_dict(sparse_8bit_lt_dict, orient='index')

# Post processing
# pt_df
# sparse_8bit_nncfkd_df
# sparse_8bit_bt_df
# sparse_8bit_lt_df

pt_df['global_step'] = 0
pt_df['category'] = "post-training sparsity"
pt_df.loc[pt_df.precision == '8bit', 'category'] = "post-training sparsity + 8bit"

sparse_8bit_nncfkd_df['category'] = "sparsity + 8bit + nncfkd"
sparse_8bit_nncfkd_df['precision'] = '8bit'

sparse_8bit_bt_df['category'] = "sparsity + 8bit + bt"
sparse_8bit_bt_df['precision'] = '8bit'

sparse_8bit_lt_df['category'] = "sparsity + 8bit + lt"
sparse_8bit_lt_df['precision'] = '8bit'

df = pd.concat([pt_df, sparse_8bit_nncfkd_df, sparse_8bit_bt_df, sparse_8bit_lt_df])

# pruneofa 90% nnz = 8493502
total_linear_params_structured_pruned_model = 20183040
total_linear_params_bert_base_model = 84934656

df=df.rename(columns={'sparsity':'unstructured_sparsity'})

df['unstructured_density'] = 100-df['unstructured_sparsity']

df['linear_density'] = ( (df['unstructured_density']*total_linear_params_structured_pruned_model/100)/total_linear_params_bert_base_model ) * 100
df['linear_sparsity'] = 100 - df['linear_density']

print('-'*100)
# official block pruning hybrid-filled-lt pareto
#-----------------------------------------------
pth='blk_pruning_official_ckpt_analysis_squadv1.json'

with open(pth, 'r') as f:
    d = json.load(f)

hybrid_filled_lt_ckpts = OrderedDict()
for k, v in d['checkpoints'].items():
    if 'sparse_args' in v:
        if 'distil_teacher_name_or_path' in v['sparse_args']:
            if v['sparse_args']['distil_teacher_name_or_path'] == 'bert-large-uncased-whole-word-masking-finetuned-squad':
                if v['sparse_args']['final_finetune'] > 0:
                    if 'fine_tuned_large_regu_' not in k:
                        hybrid_filled_lt_ckpts[k]=v
                        # print(v['sparse_args']['final_finetune'], v['sparse_args']['distil_teacher_name_or_path'])
                        # print(k.split('/')[-1])
                        # print(v['speedup'], v['stats']['total_sparsity'], v['stats']['linear_sparsity'], v['eval_metrics'])

count = Counter(list(map(lambda x: '/'.join(x.split("/")[:-1]), list(hybrid_filled_lt_ckpts.keys()))))

dictlist=[]

for k, v in hybrid_filled_lt_ckpts.items():
    entry = OrderedDict()
    entry['ckpt'] = os.path.basename(k)
    entry['root'] = os.path.dirname(k)
    entry['f1'] = v['eval_metrics']['f1']
    entry['em'] = v['eval_metrics']['exact_match']
    entry['speedup'] = v['speedup']
    entry['total_sparsity'] = v['stats']['total_sparsity']
    entry['linear_sparsity'] = v['stats']['linear_sparsity']
    dictlist.append(entry)

published_hybrid_filled_lt_df = pd.DataFrame.from_records(dictlist)

published_hybrid_filled_lt_df = published_hybrid_filled_lt_df.sort_values('f1', ascending=False).drop_duplicates(['root'])

published_hybrid_filled_lt_df['category'] = 'structured pruning'
published_hybrid_filled_lt_df['precision'] = 'FP32'

published_hybrid_filled_lt_df['ckpt'] = list(map(lambda x: str(x)[:4]+'-hybrid-filled-lt', published_hybrid_filled_lt_df.linear_sparsity.values.tolist()))
published_hybrid_filled_lt_df = published_hybrid_filled_lt_df.set_index('ckpt').drop(columns=['root', 'total_sparsity'])

published_hybrid_filled_lt_df['linear_density'] = 100-published_hybrid_filled_lt_df['linear_sparsity']
published_hybrid_filled_lt_df['unstructured_sparsity'] = 0.0
published_hybrid_filled_lt_df['unstructured_density'] = 100

# df.to_csv("tld-poc-runs.csv", index=False)
# published_hybrid_filled_lt_df.to_csv("official_hybrid_filled_lt.csv", index=False)

print('-'*100)
df = pd.concat([df, published_hybrid_filled_lt_df]).drop(columns=['global_step'])

# bert-base-squadv1 baseline
baseline_f1 = 88.5
baseline_em = 80.8

df['delta_f1']=(df['f1']-baseline_f1)/baseline_f1
df['delta_em']=(df['em']-baseline_em)/baseline_em

df['delta_f1 (%)']=df['delta_f1']*100
df['delta_em (%)']=df['delta_em']*100

df.to_csv("tld-poc-structured-unstructured-sparsity.csv", index=False)
# df[df.delta_f1 > -0.011]
print("")