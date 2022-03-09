import inspect
import types
from collections import OrderedDict

from tensorflow.keras import layers as L
from tensorflow.keras import Input

def create_init(base_cls):
    """Decorator to dynamically create init function based on wrapped `base_cls`."""

    def _get_params(func):
        params_dict = OrderedDict()
        for p in inspect.signature(func).parameters.values():
            params_dict[p.name] = p.default
        return params_dict

    def unified_init(init_func):
        # Get the dict of parameters and their values for base init and new init
        base_params = _get_params(base_cls.__init__)
        params = base_params.copy()
        params.update(_get_params(init_func))
        if 'kwargs' in params:
            params.move_to_end('kwargs')

        # Construct new __init__ function source by adding proper function definition and super() call
        init_src = inspect.getsource(init_func).split('\n') # init_src[0] is the decorator, init_src[1] is def __init__...
        init_src = init_src[1:-1] # Get rid of the decorator part and extra newline
        def_indent = len(init_src[0]) - len(init_src[0].lstrip(' '))
        src_indent = len(init_src[1]) - len(init_src[1].lstrip(' '))
        param_str = []
        for param_name, param_value in params.items():
            if param_name.startswith('kwargs'):
                param_str.append('**kwargs')
            elif param_value != inspect.Parameter.empty:
                param_str.append(f'{param_name}={param_value}')
            else:
                param_str.append(param_name)

        super_str = []
        for param_name, param_value in base_params.items():
            if param_name == 'self':
                continue
            elif param_name.startswith('kwargs'):
                super_str.append('**kwargs')
            elif param_value != inspect.Parameter.empty:
                super_str.append(f'{param_name}={param_name}')
            else:
                super_str.append(f'{param_name}')

        class_name = base_cls.__name__ # TODO A check should exist to make sure this matches the name of the wrapper

        init_src[0] = ' ' * def_indent + 'def __init__(' + ', '.join(param_str) + '):'
        super_src = ' ' * src_indent + f'super({class_name}, self).__init__(' + ', '.join(super_str) + ')'
        init_src.insert(1, super_src)

        source = ''
        for src_line in init_src:
            source += src_line[def_indent:] + '\n'

        # Compile function code from string
        code = compile(source, '<string>', 'exec')

        # Prepare default argument values (foo='bar')
        defaults = []
        for param_name, param_value in params.items():
            if param_name in ['self', 'kwargs']:
                continue
            elif param_value != inspect.Parameter.empty:
                defaults.append(param_value)
            else:
                defaults.append(None)

        # This is a bit weird, but we need to pass a different handle to the CodeType contained
        # within the `code` object, so let's get it's index (usually it's '2').
        code_ptr = 0
        for i, obj in enumerate(code.co_consts):
            if isinstance(obj, types.CodeType):
                code_ptr = i
                break

        return types.FunctionType(code.co_consts[code_ptr], init_func.__globals__, name='__init__', argdefs=tuple(defaults))

    return unified_init

#region Core layers

class Dense(L.Dense):
    @create_init(L.Dense)
    def __init__(self, strategy='latency', weight_t=None, bias_t=None, result_t=None, accum_t=None, skip_wrapping=False):
        self.strategy = strategy
        self.weight_t = weight_t
        self.bias_t = bias_t
        self.result_t = result_t
        self.accum_t = accum_t
        self.skip_wrapping = skip_wrapping

        assert self.use_bias == True

    def get_config(self):
        config = {
            'strategy': self.strategy,
            'weight_t': self.weight_t,
            'bias_t': self.bias_t,
            'result_t': self.result_t,
            'accum_t': self.accum_t,
            'skip_wrapping': self.skip_wrapping,
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BatchNormalization(L.BatchNormalization):
    @create_init(L.BatchNormalization)
    def __init__(self, scale_t=None, bias_t=None, result_t=None, skip_wrapping=False):
        self.scale_t = scale_t
        self.bias_t = bias_t
        self.result_t = result_t
        assert self.renorm == False
        assert self.virtual_batch_size == None
        assert self.adjustment == None
        self.fused = False
        self.skip_wrapping = skip_wrapping

    def get_config(self):
        config = {
            'scale_t': self.scale_t,
            'bias_t': self.bias_t,
            'result_t': self.result_t,
            'skip_wrapping': self.skip_wrapping,
        }
        base_config = super(BatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#endregion

#region Convolutional layers

class Conv2D(L.Conv2D):
    @create_init(L.Conv2D)
    def __init__(self, strategy='latency', implementation=None, weight_t=None, bias_t=None, result_t=None, accum_t=None, skip_wrapping=False):
        self.strategy = strategy
        self.implementation = implementation
        self.weight_t = weight_t
        self.bias_t = bias_t
        self.result_t = result_t
        self.accum_t = accum_t
        self.skip_wrapping = skip_wrapping

        assert self.data_format == 'channels_last'
        assert self.use_bias == True
        assert self.groups == 1
        assert self.dilation_rate == (1, 1)
        #TODO padding='same' will trip this up if using io_stream

#endregion

#region Pooling layers

class MaxPooling2D(L.MaxPooling2D):
    @create_init(L.MaxPooling2D)
    def __init__(self, implementation=None, result_t=None, accum_t=None, skip_wrapping=False):
        self.implementation = implementation
        self.result_t = result_t
        self.accum_t = accum_t
        self.skip_wrapping = skip_wrapping

        assert self.data_format == 'channels_last'

    def get_config(self):
        config = {
            'implementation': self.implementation,
            'result_t': self.result_t,
            'accum_t': self.accum_t,
            'skip_wrapping': self.skip_wrapping,
        }
        base_config = super(MaxPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AveragePooling2D(L.AveragePooling2D):
    @create_init(L.AveragePooling2D)
    def __init__(self, implementation=None, result_t=None, accum_t=None, skip_wrapping=False):
        self.implementation = implementation
        self.result_t = result_t
        self.accum_t = accum_t
        self.skip_wrapping = skip_wrapping

        assert self.data_format == 'channels_last'

    def get_config(self):
        config = {
            'implementation': self.implementation,
            'result_t': self.result_t,
            'accum_t': self.accum_t,
            'skip_wrapping': self.skip_wrapping,
        }
        base_config = super(AveragePooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

MaxPool2D = MaxPooling2D
AvgPool2D = AveragePooling2D

#endregion

#region Activations

class Softmax(L.Softmax):
    @create_init(L.Softmax)
    def __init__(self, strategy='latency', exp_table_t=None, inv_table_t=None, skip_wrapping=False):
        self.strategy = strategy
        self.exp_table_t = exp_table_t
        self.inv_table_t = inv_table_t
        self.skip_wrapping = skip_wrapping

    def get_config(self):
        config = {
            'strategy': self.strategy,
            'exp_table_t': self.exp_table_t,
            'inv_table_t': self.inv_table_t,
            'skip_wrapping': self.skip_wrapping,
        }
        base_config = super(Softmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ReLU(L.ReLU):
    @create_init(L.ReLU)
    def __init__(self, table_t=None, table_size=None, skip_wrapping=False):
        self.table_t = table_t
        self.table_size = table_size
        self.skip_wrapping = skip_wrapping

    def get_config(self):
        config = {
            'table_t': self.table_t,
            'table_size': self.table_size,
            'skip_wrapping': self.skip_wrapping,
        }
        base_config = super(ReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class LeakyReLU(L.LeakyReLU):
    @create_init(L.LeakyReLU)
    def __init__(self, table_t=None, table_size=None, skip_wrapping=False):
        self.table_t = table_t
        self.table_size = table_size
        self.skip_wrapping = skip_wrapping

    def get_config(self):
        config = {
            'table_t': self.table_t,
            'table_size': self.table_size,
            'skip_wrapping': self.skip_wrapping,
        }
        base_config = super(LeakyReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#endregion

#region Reshaping layers

class Flatten(L.Flatten):
    @create_init(L.Flatten)
    def __init__(self, skip_wrapping=False):
        self.skip_wrapping = skip_wrapping
        assert self.data_format == 'channels_last'

    def get_config(self):
        config = {
            'skip_wrapping': self.skip_wrapping,
        }
        base_config = super(Flatten, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Reshape(L.Reshape):
    pass

class UpSampling2D(L.UpSampling2D):
    @create_init(L.UpSampling2D)
    def __init__(self, skip_wrapping=False):
        self.skip_wrapping = skip_wrapping
        assert self.data_format == 'channels_last'
        assert self.interpolation == 'nearest'

    def get_config(self):
        config = {
            'skip_wrapping': self.skip_wrapping,
        }
        base_config = super(UpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ZeroPadding2D(L.ZeroPadding2D):
    @create_init(L.ZeroPadding2D)
    def __init__(self, skip_wrapping=False):
        self.skip_wrapping = skip_wrapping
        assert self.data_format == 'channels_last'

    def get_config(self):
        config = {
            'skip_wrapping': self.skip_wrapping,
        }
        base_config = super(ZeroPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#endregion