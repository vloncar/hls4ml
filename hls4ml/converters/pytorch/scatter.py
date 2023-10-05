from hls4ml.converters.pytorch_to_hls import pytorch_handler

scatter_ops = ['scatter_add', 'scatter_sum', 'scatter_max', 'scatter_mean', 'scatter_min', 'scatter_mul']


@pytorch_handler(*scatter_ops)
def parse_scatter_op(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation in scatter_ops
    assert input_names[0] == node.args[0].name

    # TODO Add support for parsing the base function, e.g., scatter(..., reduce='sum')

    layer = {}
    layer['name'] = layer_name
    layer['inputs'] = list(input_names)
    layer['class_name'] = 'Scatter'

    layer['scatter_op'] = operation.replace('scatter_', '')
    if layer['scatter_op'] == 'add':
        layer['scatter_op'] = 'sum'

    if len(node.args) > 2:
        layer['dim'] = node.args[2]
    else:
        layer['dim'] = node.kwargs.get('dim', -1)

    # Shape of the output can be extracted from "out" parameter, or from "dim_size"
    if len(node.args) > 3:
        # "out" is specified as a positional argument
        out_tensor = node.args[3]
        layer['tensor_shape'] = input_shapes[input_names.index(out_tensor.name)].copy()
        layer['init_output'] = False
    else:
        # "out" was either specified as a keyword argument or not at all
        out_tensor = node.kwargs.get('out', None)
        if out_tensor is None:
            # if present, "dim_size" will decide the output shape (again, it could be positional or keyword)
            dim_size = None
            if len(node.args) > 4:
                dim_size = node.args[4]
            else:
                dim_size = node.kwargs.get('dim_size', None)
            if dim_size is None:
                # neither "out" nor "dim_size" are specified, tensor will have minimal size
                layer['tensor_shape'] = []
            else:
                layer['tensor_shape'] = input_shapes[0].copy()
                layer['tensor_shape'][layer['dim']] = dim_size

            layer['init_output'] = True
        else:
            layer['tensor_shape'] = input_shapes[input_names.index(out_tensor.name)].copy()
            layer['init_output'] = False

    output_shape = layer['tensor_shape'].copy()

    if layer['tensor_shape'][0] is None:
        layer['tensor_shape'].pop(0)

    return layer, output_shape
