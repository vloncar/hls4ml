import numpy as np

from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Gather, GetItem

# GetItem templates

getitem_config_template = """struct config{index} : nnet::getitem_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned item_index = {item_index};
}};\n"""

getitem_function_template = 'nnet::getitem<{input_t}, {config}>({input}, {output});'

getitem_include_list = ['nnet_utils/nnet_gather.h']


class GetItemConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(GetItem)
        self.template = getitem_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['n_in'] = np.prod(node.get_input_variable().shape)
        params['n_out'] = np.prod(node.get_output_variable().shape)

        return self.template.format(**params)


class GetItemFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(GetItem, include_header=getitem_include_list)
        self.template = getitem_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


# Gather templates

gather_1d_config_template = """struct config{index} : nnet::gather_config_1d {{
    static const unsigned n_in = {n_in};
    static const unsigned n_indices = {n_indices};
    static const unsigned n_out = n_indices;
}};\n"""

gather_2d_config_template = """struct config{index} : nnet::gather_config_2d {{
    static const unsigned n_in_0 = {n_in_0};
    static const unsigned n_in_1 = {n_in_1};
    static const unsigned n_indices = {n_indices};
}};\n"""


gather_3d_config_template = """struct config{index} : nnet::gather_config_3d {{
    static const unsigned n_in_0 = {n_in_0};
    static const unsigned n_in_1 = {n_in_1};
    static const unsigned n_in_2 = {n_in_2};
    static const unsigned n_indices = {n_indices};
    
}};\n"""


gather_function_template = 'nnet::gather_{rank}d<{input_t}, {index_t}, {config}>({input}, {indices}, {output});'

gather_include_list = ['nnet_utils/nnet_gather.h']


class GatherConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Gather)
        self.templates = {
            1: gather_1d_config_template,
            2: gather_2d_config_template,
            3: gather_3d_config_template,
        }

    def format(self, node):
        params = self._default_config_params(node)
        params['n_in'] = np.prod(node.get_input_variable(node.inputs[0]).shape[1:])

        params['n_indices'] = node.get_input_variable(node.inputs[1]).shape[0]

        rank = len(node.get_input_variable(node.inputs[0]).shape)
        if rank == 1:
            params['n_in'] = node.get_input_variable(node.inputs[0]).size()

            return self.templates[1].format(**params)
        elif rank == 2:
            params['n_in_0'] = node.get_input_variable(node.inputs[0]).shape[0]
            params['n_in_1'] = node.get_input_variable(node.inputs[0]).shape[1]
            
            return self.templates[2].format(**params)
        elif rank == 3:
            params['n_in_0'] = node.get_input_variable(node.inputs[0]).shape[0]
            params['n_in_1'] = node.get_input_variable(node.inputs[0]).shape[1]
            params['n_in_2'] = node.get_input_variable(node.inputs[0]).shape[2]

            return self.templates[3].format(**params)
        else:
            return None


class GatherFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Gather, include_header=gather_include_list)
        self.template = gather_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['index_t'] = node.get_input_variable(node.inputs[1]).type.name
        params['indices'] = node.get_input_variable(node.inputs[1]).name

        params['rank'] = len(node.get_input_variable(node.inputs[0]).shape)

        return self.template.format(**params)
