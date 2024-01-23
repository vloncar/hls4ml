from hls4ml.converters.keras_to_hls import keras_handler, parse_default_keras_layer

merge_layers = ['Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum', 'Concatenate', 'Dot']


def _compute_merge_output_shape(shape1, shape2):
    # Adapted from Keras's source code: keras/layers/merging/base_merge.py
    if len(shape1) < len(shape2):
        return _compute_merge_output_shape(shape2, shape1)

    output_shape = list(shape1[: -len(shape2)])
    for i, j in zip(shape1[-len(shape2) :], shape2):
        if i is None or j is None:
            output_shape.append(None)
        elif i == 1:
            output_shape.append(j)
        elif j == 1:
            output_shape.append(i)
        else:
            if i != j:
                raise ValueError(f'Inputs have incompatible shapes. Received shapes {shape1} and {shape2}')
            output_shape.append(i)
    return list(output_shape)


@keras_handler(*merge_layers)
def parse_merge_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer['class_name'] in merge_layers

    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['op'] = layer['class_name'].lower()

    if layer['class_name'] == 'Concatenate':
        rank = len(input_shapes[0][1:])
        if rank > 3:
            raise Exception('ERROR: Concatenation of tensors with rank > 3 is not yet supported.')
        layer['op'] = layer['class_name'].lower() + f'{rank}d'
        layer['axis'] = keras_layer['config']['axis']
        output_shape = input_shapes[0][:]
        output_shape[layer['axis']] += input_shapes[1][layer['axis']]
    elif layer['class_name'] == 'Dot':
        rank = len(input_shapes[0][1:])
        if rank > 1:
            raise Exception('ERROR: Dot of tensors with rank > 1 is not yet supported.')
        layer['op'] = layer['class_name'].lower() + f'{rank}d'
        output_shape = input_shapes[0][:]
    else:
        layer['class_name'] = 'Merge'
        output_shape = _compute_merge_output_shape(input_shapes[0], input_shapes[1])

    if len(layer['inputs']) > 2:
        raise Exception('ERROR: Merging more than two tensors is not yet supported.')

    return layer, output_shape
