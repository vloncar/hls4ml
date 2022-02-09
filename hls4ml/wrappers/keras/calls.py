
def call_dense(self, inputs):
    outputs = self.op_func(inputs, self.kernel, self.bias)
    if self.act_func is not None:
        outputs = self.act_func(outputs)
    
    return outputs

def call_activation(self, inputs):
    return self.op_func(inputs)

def call_param_activation(self, inputs):
    # self.alpha will be np.ndarray (of size 1), so convert it to scalar with item()
    return self.op_func(inputs, alpha=self.alpha.item())

_call_map = {
    'dense': call_dense,
    'relu': call_activation,
    'leakyrelu': call_param_activation,
    'softmax': call_activation,
}

def lookup_call_wrapper(keras_class):
    call_func = _call_map.get(keras_class.lower(), None)

    if call_func is None:
        raise Exception(f'Cannot find __call__ wrapper for: {keras_class}.')

    return call_func