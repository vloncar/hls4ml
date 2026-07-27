from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Activation, Dense

# Weave currently supports Dense (via the BLAS-style outer-product kernel) plus elementwise
# activations (ReLU / linear / etc.). Templates for other layers are added as those features land.

# Dense templates

# Only the fields WeaveDense (and the io_stream dense wrapper) actually read are emitted.
# `strategy` is retained deliberately: it must be `resource` so the io_stream wrapper does NOT wrap the
# kernel in a forced PIPELINE (which would fully unroll it). n_in/n_out/par_entries/accum_t/bias_t/
# weight_t/product/kernel drive the kernel; reuse_factor/n_zeros/multiplier_limit/store_weights_in_bram/
# index_t/io_type are inherited from nnet::dense_config and unused here, so they are not emitted.
dense_config_template = """struct config{index} : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned par_entries = {par_entries};
    static const unsigned activation = nnet::{activation};
    static const unsigned table_size = {table_size};
    typedef {table_t} table_t;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = {dense_function}<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

dense_function_template = 'nnet::dense<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
# Fused-region forms (see backends/weave/passes/fuse_dense.py). Same (input, output, w, b) arg order;
# the difference is which side is an hls::stream.
dot_function_template = 'nnet::weave_dot<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
axpy_function_template = 'nnet::weave_axpy<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'

dense_include_list = ['nnet_utils/nnet_dense.h', 'nnet_utils/nnet_dense_stream.h', 'nnet_utils/nnet_dense_weave.h']


class DenseConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Dense)
        self.template = dense_config_template

    # Elementwise activation folded into the kernel output stage by weave_fold_activation.
    _ACT_ENUM = {None: 'WEAVE_LINEAR', 'linear': 'WEAVE_LINEAR', 'relu': 'WEAVE_RELU',
                 'tanh': 'WEAVE_TANH', 'sigmoid': 'WEAVE_SIGMOID'}

    def format(self, node):
        params = self._default_config_params(node)
        params['par_entries'] = node.get_attr('par_entries', 1)
        params['activation'] = self._ACT_ENUM.get(node.get_attr('weave_act'), 'WEAVE_LINEAR')
        # table only used by tanh/sigmoid; defaults match hls4ml's activation defaults (DCE'd otherwise)
        params['table_size'] = node.get_attr('weave_table_size') or 1024
        params['table_t'] = node.get_attr('weave_table_t') or 'ap_fixed<18,8>'
        params['product_type'] = get_backend('Weave').product_type(
            node.get_input_variable().type.precision, node.get_weights('weight').type.precision
        )

        # Weave always routes Dense through its own BLAS-style outer-product kernel.
        params['dense_function'] = 'nnet::WeaveDense'

        return self.template.format(**params)


class DenseFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Dense, include_header=dense_include_list)
        self.template = dense_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name

        # Pick the kernel matching the form the fusion planner assigned to this layer.
        template = {
            'dot': dot_function_template,
            'axpy': axpy_function_template,
        }.get(node.get_attr('weave_form'), self.template)

        return template.format(**params)


# Activation templates

activ_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef {table_t.name} table_t;
}};\n"""

activ_function_template = 'nnet::{activation}<{input_t}, {output_t}, {config}>({input}, {output});'

activ_include_list = ['nnet_utils/nnet_activation.h', 'nnet_utils/nnet_activation_stream.h']


class ActivationConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Activation)
        self.template = activ_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['type'] = node.get_attr('activation')

        return self.template.format(**params)


class ActivationFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Activation, include_header=activ_include_list)
        self.template = activ_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['activation'] = node.get_attr('activation').lower()
        params['config'] = '{}_config{}'.format(node.get_attr('activation'), node.index)

        return self.template.format(**params)
