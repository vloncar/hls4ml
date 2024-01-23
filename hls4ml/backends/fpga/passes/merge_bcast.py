from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Layer, Merge, register_layer
from hls4ml.model.optimizer import OptimizerPass


class MergeBroadcast(Layer):
    '''Inserted before the Merge layer that requires broadcasting (i.e., one of the tensors has a dimension of 1).'''

    def initialize(self):
        shape = self.get_attr('target_shape')
        dim_names = [f'OUT_BCAST_{self.index}_{i}' for i in range(len(shape))]
        self.add_output_variable(shape, dim_names)


merge_bcast_include_list = ['nnet_utils/nnet_merge.h']
merge_bcast_function_template = 'nnet::merge_bcast<{input_t}, {output_t}, {config}>({input}, {output});'
merge_bcast_config_template = """struct config{index} : nnet::merge_bcast_config {{
    static const unsigned n_elem1_0 = {n_elem1_0};
    static const unsigned n_elem1_1 = {n_elem1_1};
    static const unsigned n_elem1_2 = {n_elem1_2};
    static const unsigned n_elem2_0 = {n_elem2_0};
    static const unsigned n_elem2_1 = {n_elem2_1};
    static const unsigned n_elem2_2 = {n_elem2_2};

    static const int axis = {axis};
}};\n"""


class MergeBroadcastFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(MergeBroadcast, include_header=merge_bcast_include_list)
        self.template = merge_bcast_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


class MergeBroadcastConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(MergeBroadcast)
        self.template = merge_bcast_config_template

    def format(self, node):
        params = self._default_config_params(node)
        for i in range(3):
            params.setdefault(f'n_elem1_{i}', 1)
            params.setdefault(f'n_elem2_{i}', 1)
        inp = node.get_input_variable()
        out = node.get_output_variable()
        rank = len(out.shape)
        dim_offset = 3 - rank
        params['axis'] += dim_offset
        for i, (s1, s2) in enumerate(zip(inp.shape, out.shape)):
            idx = i + dim_offset
            params[f'n_elem1_{idx}'] = s1
            params[f'n_elem2_{idx}'] = s2

        return self.template.format(**params)


def register_merge_bcast(backend):
    # Register the layer types to the layer map
    register_layer('MergeBroadcast', MergeBroadcast)

    # Register the optimization passes
    backend.register_pass('merge_broadcast', MergeBroadcastDim)

    # Register template passes
    backend.register_template(MergeBroadcastFunctionTemplate)
    backend.register_template(MergeBroadcastConfigTemplate)


class MergeBroadcastDim(OptimizerPass):
    '''Inserts a layer that will broadcast a tensor before the Merge layer.'''

    def match(self, node):
        if isinstance(node, Merge):
            inp1_shape = node.get_input_variable(node.inputs[0]).shape
            inp2_shape = node.get_input_variable(node.inputs[1]).shape
            for i, j in zip(inp1_shape, inp2_shape):
                if i != j and (i == 1 or j == 1):
                    return True

        return False

    def transform(self, model, node):
        inp1_shape = node.get_input_variable(node.inputs[0]).shape
        inp2_shape = node.get_input_variable(node.inputs[1]).shape
        for d, (i, j) in enumerate(zip(inp1_shape, inp2_shape)):
            if i != j:
                axis = d
                if i == 1:
                    idx = 0
                    bcast_inp = node.inputs[idx]
                    target_shape = inp1_shape.copy()
                    target_shape[d] = j
                else:
                    idx = 1
                    bcast_inp = node.inputs[idx]
                    target_shape = inp2_shape.copy()
                    target_shape[d] = i
                # We break because there may be another dimension of size 1,
                # for that we need another broadcast (another run of the optimizer)
                break

        attrs = {
            'target_shape': target_shape,
            'axis': axis,
        }

        bcast_layer = model.make_node(MergeBroadcast, node.inputs[idx] + '_bcast', attrs, [bcast_inp].copy())
        model.insert_node(bcast_layer, before=node, input_idx=idx)

        return True
