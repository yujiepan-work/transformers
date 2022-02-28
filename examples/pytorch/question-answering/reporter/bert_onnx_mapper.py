import onnx
from collections import OrderedDict
from onnx import numpy_helper

from onnx.parser import parse_model 
# print(onnx.helper.printable_graph(onnx_model.graph))

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.float_format', '{:20,.2f}'.format)
pd.set_option('display.max_colwidth', None)

def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

def find_node_by_output(output :str, graph):
    # assuming output node is a singleton
    for node in graph.node:
        if output in node.output:
            break
    return node

def find_node(node_name, graph):
    for node in graph.node:
        if node_name == node.name:
            break
    return node

def find_initializer_node(init_node_name, graph):
    for init in graph.initializer:
        if init.name == init_node_name:
            break
    return init

class bert_onnx_mapper:
    def __init__(self, onnx_pth, variant=None):
        self.onnx_pth = onnx_pth
        self.variant = None
        onnx_model = onnx.load(onnx_pth)
        self.onnx_model = onnx_model

        if variant == 'nncf-quantized':
            quantized_tensor_nodes = OrderedDict()
            for node in onnx_model.graph.node:
                if 'fakequantize' in node.name.lower():
                    # From visual inspection, node.input[0] first input appears to be the constant tensor to be quantized
                    # From NodeProto of onnx, input is a list of string where each string is the name of output node of Constant Node (this is not initializer)
                    constant_node = find_node_by_output(node.input[0], onnx_model.graph)
                    if  constant_node.op_type == 'Constant':
                        if len(constant_node.attribute[0].t.dims) > 1:
                            print(node.name, 
                                ", input[0]:", node.input[0], 
                                ", constant_node:", constant_node.name, 
                                ", constant_node_type:", constant_node.op_type, 
                                ", dims:", constant_node.attribute[0].t.dims)
                        quantized_tensor_nodes[constant_node.name] = numpy_helper.to_array(constant_node.attribute[0].t)

            for init in onnx_model.graph.initializer:
                if 'bias' in init.name and 'bert.encoder' in init.name and 'LayerNorm' not in init.name:
                    print("bias_init_node:", init.name, ", dims:", init.dims)
                    quantized_tensor_nodes[init.name] = numpy_helper.to_array(init)

            self.quantized_tensor_nodes = None
            if len(quantized_tensor_nodes) > 0:
                self.quantized_tensor_nodes = quantized_tensor_nodes
        elif variant == 'nncf-sparsified':
            tensor_nodes = OrderedDict()
            for node in onnx_model.graph.node:
                if 'add' in node.name.lower():
                    for innode in node.input:
                        if 'bias' in innode and 'LayerNorm' not in innode:
                            # print(node.name, node.input)

                            bias_init_node = find_initializer_node(node.input[0], onnx_model.graph)
                            print("bias_init_node:", bias_init_node.name, ", dims:", bias_init_node.dims)
                            tensor_nodes[bias_init_node.name] = numpy_helper.to_array(bias_init_node)

                            weight_init_node_name = node.input[0].replace('bias','weight')
                            weight_init_node = find_initializer_node(weight_init_node_name, onnx_model.graph)
                            print("weight_init_node:", weight_init_node.name, ", dims:", weight_init_node.dims)
                            tensor_nodes[weight_init_node.name] = numpy_helper.to_array(weight_init_node)
            if len(tensor_nodes) > 0:
                self.tensor_nodes = tensor_nodes

if __name__ == "__main__":
    import torch
    import os
    from sparsity_reporter import SparsityReporter
    # onnx_pth = '/tmp/vscode-runs/ssbs-feb/prune-bias/NNCFNetwork.onnx'
    # mapper = bert_onnx_mapper(onnx_pth, variant='nncf-quantized')
    # df = SparsityReporter.per_item_sparsity(mapper.quantized_tensor_nodes)

    # onnx_pth = '/tmp/vscode-runs/ssbs-feb/prune-bias/NNCFNetwork.onnx'
    # onnx_pth = '/data2/vchua/run/feb-topt/bert-squad/run27-bert-squad-nncf-mvmt-bt-20eph-r0.02-threshold-end-3eph-prune-bias-prefilled/checkpoint-35000/NNCFNetwork.onnx'
    # mapper = bert_onnx_mapper(onnx_pth, variant='nncf-sparsified')
    # df = SparsityReporter.per_item_sparsity(mapper.tensor_nodes)
    # df.to_csv("onnx_sparsity.csv", index=True)
    # with open('onnx_sparsity.md', 'w') as f:
    #     df.to_markdown(f)

    # ckptpth = '/data2/vchua/run/feb-topt/bert-squad/run27-bert-squad-nncf-mvmt-bt-20eph-r0.02-threshold-end-3eph-prune-bias-prefilled/checkpoint-60000/pytorch_model.bin'
    # df60 = SparsityReporter.per_item_sparsity(torch.load(ckptpth))

    # ckptpth = '/data2/vchua/run/feb-topt/bert-squad/run27-bert-squad-nncf-mvmt-bt-20eph-r0.02-threshold-end-3eph-prune-bias-prefilled/checkpoint-57500/pytorch_model.bin'
    # df57 = SparsityReporter.per_item_sparsity(torch.load(ckptpth))

    # ckptpth = '/data2/vchua/run/feb-topt/bert-squad/run27-bert-squad-nncf-mvmt-bt-20eph-r0.02-threshold-end-3eph-prune-bias-prefilled/checkpoint-62500/pytorch_model.bin'
    # df62 = SparsityReporter.per_item_sparsity(torch.load(ckptpth))

    ckptpth = '/data2/vchua/run/feb-topt/bert-squad/run27.fri-bert-squad-nncf-mvmt-lt-20eph-r0.02-threshold-end-3eph-prune-bias-filled/checkpoint-95000/pytorch_model.bin'
    sd = torch.load(ckptpth)
    df95 = SparsityReporter.per_item_sparsity(sd)
    sparsified_sd = OrderedDict()
    for k, v in sd.items():
        sparsified_sd[k] = v # copy every key value pair then only overriding them if needed
        if 'binary_mask' in k:
            key = k.split("pre_ops")[0]
            if 'weight' in k:
                key += 'weight'
            elif 'bias' in k:
                key += 'bias'
            else:
                raise ValueError("unexpected entry 1, pls debug")
            if key in sd:
                # print("{} exists in sd".format(key))
                print("{}\n\t{}\n".format(key, k))
                sparsified_sd[k] = v
                sparsified_sd[key] = sd[key] * v
            else:
                raise ValueError("unexpected entry 2, pls debug")

    if len(sd) != len(sparsified_sd):
        raise ValueError("#key mismatched")
    sparse_df95 = SparsityReporter.per_item_sparsity(sparsified_sd)

    sparse_ckptpth = os.path.join(os.path.dirname(ckptpth), "sparsified_pytorch_model.bin")
    torch.save(sparsified_sd, sparse_ckptpth)

    # dense_sd = torch.load("/data2/vchua/run/feb-topt/modelhub/bert-base-squadv1/pytorch_model.bin")
    dense_sd = torch.load("/data2/vchua/run/feb-topt/modelhub/bert-base-uncased-squad/pytorch_model.bin")
    sparse_hfsd = OrderedDict()
    for k, _ in dense_sd.items():
        key = 'nncf_module.' + k
        if key in sparsified_sd:
            sparse_hfsd[k] = sparsified_sd[key]
            pass
        else:
            raise ValueError("this k will be missing in sparse hfsd, pls debug")

    assert len(dense_sd) == len(sparse_hfsd), "Keys mismatch"
    df_sparse_hfsd = SparsityReporter.per_item_sparsity(sparse_hfsd)
    sparse_hfsd_ckptpth = os.path.join(os.path.dirname(ckptpth), "hf_sparse_pytorch_model.bin")
    torch.save(sparse_hfsd, sparse_hfsd_ckptpth)
    print("dummy")