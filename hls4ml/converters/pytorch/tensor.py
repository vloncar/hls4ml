from hls4ml.converters.pytorch_to_hls import pytorch_handler

tensor_like_ops = ['zeros_like', 'ones_like']


@pytorch_handler(*tensor_like_ops)
def parse_tensor_like_op(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation in tensor_like_ops
    assert input_names[0] == node.args[0].name

    layer = {}
    layer['name'] = layer_name
    layer['inputs'] = list(input_names)

    layer['class_name'] = 'FillTensor'
    layer['fill_op'] = operation.replace('_like', '')
    layer['tensor_shape'] = input_shapes[0].copy()

    output_shape = layer['tensor_shape'].copy()

    if layer['tensor_shape'][0] is None:
        layer['tensor_shape'].pop(0)

    return layer, output_shape
