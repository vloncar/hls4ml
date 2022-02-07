from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops

def dense_gradient(op, grad):
    i = math_ops.conj(op.inputs[0])
    w = math_ops.conj(op.inputs[1])

    grad_i = gen_math_ops.mat_mul(grad, w, transpose_b=True)
    grad_w = gen_math_ops.mat_mul(i, grad, transpose_a=True)

    grad_b = gen_nn_ops.bias_add_grad(out_backprop=grad, data_format='NHWC')

    return grad_i, grad_w, grad_b

relu_gradient = ops._gradient_registry.lookup('Relu')
leakyrelu_gradient = ops._gradient_registry.lookup('LeakyRelu')
selu_gradient = ops._gradient_registry.lookup('Selu')
elu_gradient = ops._gradient_registry.lookup('Elu')
sigmoid_gradient = ops._gradient_registry.lookup('Sigmoid')
tanh_gradient = ops._gradient_registry.lookup('Tanh')
softplus_gradient = ops._gradient_registry.lookup('Softplus')
softsign_gradient = ops._gradient_registry.lookup('Softsign')
softmax_gradient = ops._gradient_registry.lookup('Softmax')

_gradient_map = {
    # Layers
    'Dense': dense_gradient,
    # Activations
    'Relu': relu_gradient,
    'LeakyRelu': leakyrelu_gradient,
    'Elu': elu_gradient,
    'Selu': selu_gradient,
    'Sigmoid': sigmoid_gradient,
    'Tanh': tanh_gradient,
    'Softplus': softplus_gradient,
    'Softsign': softsign_gradient,
    'Softmax': softmax_gradient,
}

def register_gradient(wrapper_name, keras_class):
    grad_func = _gradient_map.get(keras_class, None)

    if grad_func is None:
        raise Exception(f'Cannot register gradient for: {wrapper_name} ({keras_class}). No mathing gradient function found.')

    ops._gradient_registry.register(grad_func, wrapper_name)