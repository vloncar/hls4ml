from hls4ml.converters.pytorch_to_hls import pytorch_handler


@pytorch_handler('getitem')
def parse_getitem_op(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation == 'getitem'
    assert input_names[0] == node.args[0].name

    layer = {}
    layer['name'] = layer_name
    layer['inputs'] = list(input_names)

    index_arg = node.args[1]
    if isinstance(index_arg, int):
        layer['class_name'] = 'GetItem'
        layer['item_index'] = index_arg
        output_shape = input_shapes[0].copy()
        output_shape.pop(1)
    elif isinstance(index_arg, slice):
        slice_op = index_arg
        if slice_op.start is not None or slice_op.stop is not None or slice_op.step is not None:
            raise Exception('Advanced slice operations are not yet supported.')
        else:
            # Until we can parse slice operation, assume we're taking the whole slice and represent it as a linear
            # activation (a no-op that gets eliminated)
            layer['class_name'] = 'Activation'
            layer['activation'] = 'linear'
            output_shape = input_shapes[0].copy()
    elif isinstance(index_arg, tuple):
        raise Exception('Advanced slice operations are not yet supported.')
    else:  # torch.fx.node.Node
        layer['class_name'] = 'Gather'
        assert len(input_shapes[1][1:]) == 1, 'Gather operation is only supported for indices tensor of rank 1.'
        output_shape = input_shapes[0].copy()
        output_shape[1] = input_shapes[1][-1]

    return layer, output_shape
