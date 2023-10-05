from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Scatter

# scatter config templates

scatter_1d_config_template = """struct config{index} : nnet::scatter_config_1d {{
    static const unsigned n_in = {n_in};
    static const unsigned n_index = {n_index};
    static const unsigned n_out = {n_out};
    static const bool init_output = {init_output};
    static const nnet::Scatter_Op scatter_op = nnet::Scatter{scatter_op};
}};\n"""

scatter_2d_config_template = """struct config{index} : nnet::scatter_config_2d {{
    static const unsigned n_in_0 = {n_in_0};
    static const unsigned n_in_1 = {n_in_1};
    static const unsigned n_index_0 = {n_index_0};
    static const unsigned n_index_1 = {n_index_1};
    static const unsigned n_out_0 = {n_out_0};
    static const unsigned n_out_1 = {n_out_1};
    static const unsigned dim = {dim};
    static const bool init_output = {init_output};
    static const nnet::Scatter_Op scatter_op = nnet::Scatter{scatter_op};
}};\n"""


scatter_3d_config_template = """struct config{index} : nnet::scatter_config_3d {{
    static const unsigned n_in_0 = {n_in_0};
    static const unsigned n_in_1 = {n_in_1};
    static const unsigned n_in_2 = {n_in_2};
    static const unsigned n_index_0 = {n_index_0};
    static const unsigned n_index_1 = {n_index_1};
    static const unsigned n_index_2 = {n_index_2};
    static const unsigned n_out_0 = {n_out_0};
    static const unsigned n_out_1 = {n_out_1};
    static const unsigned n_out_2 = {n_out_2};
    static const unsigned dim = {dim};
    static const bool init_output = {init_output};
    static const nnet::Scatter_Op scatter_op = nnet::Scatter{scatter_op};
}};\n"""


scatter_function_template = (
    'nnet::scatter_{rank}d<{input_t}, {index_t}, {target_t}, {output_t}, {config}>({input}, {index}, {target}, {output});'
)


scatter_include_list = ['nnet_utils/nnet_scatter.h']


class ScatterConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Scatter)
        self.templates = {
            1: scatter_1d_config_template,
            2: scatter_2d_config_template,
            3: scatter_3d_config_template,
        }

    def format(self, node):
        params = self._default_config_params(node)
        params['scatter_op'] = params['scatter_op'].capitalize()

        rank = len(node.get_input_variable(node.inputs[0]).shape)
        if rank == 1:
            params['n_in'] = node.get_input_variable(node.inputs[0]).size()
            params['n_index'] = node.get_input_variable(node.inputs[1]).size()
            params['n_out'] = node.get_output_variable().size()

            return self.templates[1].format(**params)
        elif rank == 2:
            params['n_in_0'] = node.get_input_variable(node.inputs[0]).shape[0]
            params['n_in_1'] = node.get_input_variable(node.inputs[0]).shape[1]
            index_var = node.get_input_variable(node.inputs[1])
            if len(index_var.shape) == 1:
                params['n_index_0'] = 1
                params['n_index_1'] = node.get_input_variable(node.inputs[1]).shape[0]
            elif len(index_var.shape) == 2:
                params['n_index_0'] = node.get_input_variable(node.inputs[1]).shape[0]
                params['n_index_1'] = node.get_input_variable(node.inputs[1]).shape[1]
            params['n_out_0'] = node.get_output_variable().shape[0]
            params['n_out_1'] = node.get_output_variable().shape[1]

            return self.templates[2].format(**params)
        elif rank == 3:
            params['n_in_0'] = node.get_input_variable(node.inputs[0]).shape[0]
            params['n_in_1'] = node.get_input_variable(node.inputs[0]).shape[1]
            params['n_in_2'] = node.get_input_variable(node.inputs[0]).shape[2]
            index_var = node.get_input_variable(node.inputs[1])
            if len(index_var.shape) == 1:
                params['n_index_0'] = 1
                params['n_index_1'] = 1
                params['n_index_2'] = node.get_input_variable(node.inputs[1]).shape[0]
            elif len(index_var.shape) == 2:
                params['n_index_0'] = 1
                params['n_index_1'] = node.get_input_variable(node.inputs[1]).shape[0]
                params['n_index_2'] = node.get_input_variable(node.inputs[1]).shape[1]
            elif len(index_var.shape) == 3:
                params['n_index_0'] = node.get_input_variable(node.inputs[1]).shape[0]
                params['n_index_1'] = node.get_input_variable(node.inputs[1]).shape[1]
                params['n_index_2'] = node.get_input_variable(node.inputs[1]).shape[2]
            params['n_out_0'] = node.get_output_variable().shape[0]
            params['n_out_1'] = node.get_output_variable().shape[1]
            params['n_out_2'] = node.get_output_variable().shape[2]

            return self.templates[3].format(**params)
        else:
            return None


class ScatterFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((Scatter), include_header=scatter_include_list)
        self.template = scatter_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['input_t'] = node.get_input_variable(node.inputs[0]).type.name
        params['index_t'] = node.get_input_variable(node.inputs[1]).type.name
        params['input'] = node.get_input_variable(node.inputs[0]).name
        params['index'] = node.get_input_variable(node.inputs[1]).name
        if len(node.inputs) == 3:
            params['target_t'] = node.get_input_variable(node.inputs[2]).type.name
            params['target'] = node.get_input_variable(node.inputs[2]).name
        else:
            params['target_t'] = params['input_t']  # Doesn't matter
            params['target'] = 'nullptr'

        params['rank'] = len(node.get_input_variable(node.inputs[0]).shape)

        return self.template.format(**params)
