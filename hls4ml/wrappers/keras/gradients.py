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

def bn_gradient(op, grad):
    x = op.inputs[0]
    grad_y = grad
    gamma = op.inputs[1]
    mean = op.inputs[3]
    variance = op.inputs[4]
    epsilon = op.get_attr("epsilon")
    is_training = op.get_attr("is_training")

    #TODO Handle 3D/4D/5D input with different data format based on the implementation from _BatchNormGrad (in nn_grad.py)

    if is_training:
        grad_x = gamma * math_ops.rsqrt(math_ops.reduce_variance(x) + epsilon) * \
            (grad_y - math_ops.reduce_mean(grad_y) - (x - math_ops.reduce_mean(x)) * \
            math_ops.reduce_mean(grad_y * (x - math_ops.reduce_mean(x))) / (math_ops.reduce_variance(x) + epsilon))
        grad_gamma = math_ops.reduce_sum(grad_y * (x - math_ops.reduce_mean(x)) * math_ops.rsqrt(math_ops.reduce_variance(x) + epsilon), axis=0)
    else:
        grad_x = grad_y * gamma * math_ops.rsqrt(variance + epsilon)
        grad_gamma = math_ops.reduce_sum(grad_y * (x - mean) * math_ops.rsqrt(variance + epsilon), axis=0)

    grad_beta = math_ops.reduce_sum(grad_y, axis=0)
    
    return grad_x, grad_gamma, grad_beta, None, None

def upsampling2d_gradient(op, grad):
    #TODO Handle half_pixel_centers=True for nearest neighbor implementation
    return tf_reshape_gradient(op, grad)

tf_reshape_gradient = ops._gradient_registry.lookup('Reshape')

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
    'dense': dense_gradient,
    'batchnormalization': bn_gradient,
    'upsampling2d': upsampling2d_gradient,
    # Activations
    'relu': relu_gradient,
    'leakyrelu': leakyrelu_gradient,
    'elu': elu_gradient,
    'selu': selu_gradient,
    'sigmoid': sigmoid_gradient,
    'tanh': tanh_gradient,
    'softplus': softplus_gradient,
    'softsign': softsign_gradient,
    'softmax': softmax_gradient,
}

def register_gradient(wrapper_name, keras_class):
    grad_func = _gradient_map.get(keras_class.lower(), None)

    if grad_func is None:
        raise Exception(f'Cannot register gradient for: {wrapper_name} ({keras_class}). No mathing gradient function found.')

    ops._gradient_registry.register(grad_func, wrapper_name)