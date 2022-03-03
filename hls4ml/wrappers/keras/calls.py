import tensorflow as tf
from keras.utils import control_flow_util

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

def call_batchnorm(self, inputs, training=None):
    training = self._get_training_value(training)

    inputs_dtype = inputs.dtype.base_dtype
    if inputs_dtype in (tf.float16, tf.bfloat16):
        # Do all math in float32 if given 16-bit inputs for numeric stability.
        # In particular, it's very easy for variance to overflow in float16 and
        # for safety we also choose to cast bfloat16 to float32.
        inputs = tf.cast(inputs, tf.float32)

    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.shape
    ndims = len(input_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.axis]

    # Broadcasting only necessary for single-axis batch norm where the axis is not the last dimension
    broadcast_shape = [1] * ndims
    broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

    def _broadcast(v):
        if (v is not None and len(v.shape) != ndims and reduction_axes != list(range(ndims - 1))):
            return tf.reshape(v, broadcast_shape)
        return v

    scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

    # Determine a boolean value for `training`: could be True, False, or None.
    training_value = control_flow_util.constant_value(training)
    if training_value == False:  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
        mean, variance = self.moving_mean, self.moving_variance
    else:
        # Some of the computations here are not necessary when training==False
        # but not a constant. However, this makes the code simpler.
        keep_dims = len(self.axis) > 1
        mean, variance = self._moments(
            tf.cast(inputs, self._param_dtype),
            reduction_axes,
            keep_dims=keep_dims)
  
        mean = control_flow_util.smart_cond(training,
            lambda: mean,
            lambda: tf.convert_to_tensor(self.moving_mean))
        variance = control_flow_util.smart_cond(training,
            lambda: variance,
            lambda: tf.convert_to_tensor(self.moving_variance))
  
        new_mean, new_variance = mean, variance
  
        if self._support_zero_size_input():
            # Keras assumes that batch dimension is the first dimension for Batch Normalization.
            input_batch_size = tf.shape(inputs)[0]
        else:
            input_batch_size = None
  
        def _do_update(var, value):
            """Compute the updates for mean and variance."""
            return self._assign_moving_average(var, value, self.momentum, input_batch_size)
  
        def mean_update():
            true_branch = lambda: _do_update(self.moving_mean, new_mean)
            false_branch = lambda: self.moving_mean
            return control_flow_util.smart_cond(training, true_branch, false_branch)
  
        def variance_update():
            """Update the moving variance."""
            true_branch = lambda: _do_update(self.moving_variance, new_variance)
            false_branch = lambda: self.moving_variance
            return control_flow_util.smart_cond(training, true_branch, false_branch)
  
        self.add_update(mean_update)
        self.add_update(variance_update)

    mean = tf.cast(mean, inputs.dtype)
    variance = tf.cast(variance, inputs.dtype)
    if offset is not None:
        offset = tf.cast(offset, inputs.dtype)
    if scale is not None:
        scale = tf.cast(scale, inputs.dtype)
    
    outputs = self.op_func(inputs, scale, offset, _broadcast(mean), _broadcast(variance), epsilon=self.epsilon, is_training=training)
    
    if inputs_dtype in (tf.float16, tf.bfloat16):
        outputs = tf.cast(outputs, inputs_dtype)

    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)
    return outputs

def call_upsampling2d(self, inputs):
    return self.op_func(inputs, factor=self.size)

_call_map = {
    'dense': call_dense,
    'batchnormalization': call_batchnorm,
    'upsampling2d': call_upsampling2d,
    'relu': call_activation,
    'leakyrelu': call_param_activation,
    'softmax': call_activation,
}

def lookup_call_wrapper(keras_class):
    call_func = _call_map.get(keras_class.lower(), None)

    if call_func is None:
        raise Exception(f'Cannot find __call__ wrapper for: {keras_class}.')

    return call_func