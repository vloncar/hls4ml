from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import FillTensor

fill_config_template = """struct config{index} : nnet::fill_config {{
    static const unsigned n_in = {n_in};
    static const unsigned fill_value = {fill_value};
}};\n"""

fill_function_template = 'nnet::fill_tensor<{output_t}, {config}>({output});'

fill_include_list = ['nnet_utils/nnet_array.h']


class FillTensorConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(FillTensor)
        self.template = fill_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['n_in'] = node.get_output_variable().size_cpp()
        fill_op = node.get_attr('fill_op')
        if fill_op == 'ones':
            params['fill_value'] = 1
        elif fill_op == 'zeros':
            params['fill_value'] = 0
        else:
            raise Exception('Function "fill_tensor" currently only supports "ones" and "zeros".')

        return self.template.format(**params)


class FillTensorFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(FillTensor, include_header=fill_include_list)
        self.template = fill_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)
