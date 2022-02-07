
def call_dense(self, inputs):
    outputs = self.op_func(inputs, self.kernel, self.bias)
    if self.act_func is not None:
        outputs = self.act_func(outputs)
    
    return outputs

def call_activation(self, inputs):
    return self.op_func(inputs)


_call_map = {
    'Dense': call_dense,
    'Softmax': call_activation
}

def lookup_call_wrapper(keras_class):
    call_func = _call_map.get(keras_class, None)

    if call_func is None:
        raise Exception(f'Cannot find __call__ wrapper for: {keras_class}.')

    return call_func