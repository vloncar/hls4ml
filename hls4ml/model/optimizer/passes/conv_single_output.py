import numpy as np
import re

from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.hls_model import Conv1D, Conv2D, register_layer
from hls4ml.templates import templates

class SingleOutputConv1D(Conv1D):
    ''' Optimized Conv1D implementation for kernel_size = input_size resulting in single output pixel. '''

    # Nothing to do, will pick up function and config from class name
    pass

class SingleOutputConv2D(Conv2D):
    ''' Optimized Conv2D implementation for kernel_size = input_size resulting in single output pixel. '''

    # Nothing to do, will pick up function and config from class name
    pass

single_out_conv1d_function_template = 'nnet::single_output_conv_1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
single_out_conv2d_function_template = 'nnet::single_output_conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'

single_out_conv1d_include_list = ['nnet_utils/nnet_conv1d.h', 'nnet_utils/nnet_conv1d_stream.h']
single_out_conv2d_include_list = ['nnet_utils/nnet_conv2d.h', 'nnet_utils/nnet_conv2d_stream.h']

# Register the layer types to the layer map
register_layer('SingleOutputConv1D', SingleOutputConv1D)
register_layer('SingleOutputConv2D', SingleOutputConv2D)

# Register the templates for config and function
templates.get_backend('Vivado').register_templates(
    'SingleOutputConv1D',
    single_out_conv1d_function_template,
    templates.get_backend('Vivado').get_config_template('Conv1D'),
    single_out_conv1d_include_list
)

templates.get_backend('Vivado').register_templates(
    'SingleOutputConv2D',
    single_out_conv2d_function_template,
    templates.get_backend('Vivado').get_config_template('Conv2D'),
    single_out_conv2d_include_list
)

class OptimizeSingleOutConv(OptimizerPass):
    def match(self, node):
        return node.__class__.__name__ in ['Conv1D', 'Conv2D'] and \
            node.get_attr('filt_height', 1) == node.get_attr('in_height', 1) and \
            node.get_attr('filt_width') == node.get_attr('in_width') and \
            node.get_attr('out_height', 1) == 1 and node.get_attr('out_width') == 1

    def transform(self, model, node):
        dim = node.__class__.__name__[-2:] # '1D' or '2D'
        pw_node = model.make_node('SingleOutputConv' + dim, node.name, node.attributes.copy(), node.inputs.copy())
        model.replace_node(node, pw_node)
        
        return True
