import onnx
from collections import OrderedDict
from onnx import numpy_helper

from onnx.parser import parse_model 
# print(onnx.helper.printable_graph(onnx_model.graph))

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
                    # From NodeProto of onnx, input is a list of string where string is the name of output node of Constant Node (this is not initializer)
                    constant_node = find_node_by_output(node.input[0], onnx_model.graph)
                    if  constant_node.op_type == 'Constant':
                        if len(constant_node.attribute[0].t.dims) > 1:
                            print(node.name, 
                                ", input[0]:", node.input[0], 
                                ", constant_node:", constant_node.name, 
                                ", constant_node_type:", constant_node.op_type, 
                                ", dims:", constant_node.attribute[0].t.dims)
                        quantized_tensor_nodes[constant_node.name] = numpy_helper.to_array(constant_node.attribute[0].t)

            self.quantized_tensor_nodes = None
            if len(quantized_tensor_nodes) > 0:
                self.quantized_tensor_nodes = quantized_tensor_nodes
        
if __name__ == "__main__":
    onnx_pth = '/tmp/vscode-runs/tld-poc/ptq-eval-squad-v1/test.onnx'
    mapper = bert_onnx_mapper(onnx_pth, variant='nncf-quantized')

    from sparsity_reporter import SparsityReporter

    df = SparsityReporter.per_item_sparsity(mapper.quantized_tensor_nodes)

    print("dummy")