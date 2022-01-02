
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch
import torch.nn as nn


class SparsityReporter():
    def __init__(
        self,
        model: nn.Module) -> None:
        
        self.model = model
        self.sparsity_df = self._get_layer_wise_sparsity()

    @staticmethod
    def calc_sparsity(tensor):
        rate = 1-(tensor.count_nonzero()/tensor.numel())
        if isinstance(rate, torch.Tensor):
            return rate.item()
        return rate

    @staticmethod
    def per_item_sparsity(state_dict):
        dlist=[]
        for key, param in state_dict.items():
            l = OrderedDict()
            l['layer_id'] = key
            l['shape'] = list(param.shape)
            l['nparam'] = np.prod(l['shape'])
            l['nnz'] = param.count_nonzero().item()
            l['sparsity'] = SparsityReporter.calc_sparsity(param)
            dlist.append(l)
        df = pd.DataFrame.from_dict(dlist)
        return df

    def _get_layer_wise_sparsity(self):
        dlist=[]
        for n, m in self.model.named_modules():
            
            if hasattr(m, 'weight'):
                l = OrderedDict()
                l['layer_id'] = n
                l['layer_type'] = m.__class__.__name__
                l['param_type'] = 'weight'
                l['shape'] = list(m.weight.shape)
                l['nparam'] = np.prod(l['shape'])
                l['nnz'] = m.weight.count_nonzero().item()
                l['sparsity'] = self.calc_sparsity(m.weight)
                dlist.append(l)

            if hasattr(m, 'bias'):
                l = OrderedDict()
                l['layer_id'] = n
                l['layer_type'] = m.__class__.__name__
                l['param_type'] = 'bias'
                l['shape'] = list(m.bias.shape)
                l['nparam'] = np.prod(l['shape'])
                l['nnz'] = m.bias.count_nonzero().item()
                l['sparsity'] = self.calc_sparsity(m.bias)
                dlist.append(l)
                
        df = pd.DataFrame.from_dict(dlist)
        return df
