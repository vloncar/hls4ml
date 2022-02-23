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

        class_name = base_cls.__name__

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

class Dense(L.Dense):

    @create_init(L.Dense)
    def __init__(self, strategy='latency', weight_t=None, bias_t=None, result_t=None, accum_t=None):
        self.strategy = strategy
        self.weight_t = weight_t
        self.bias_t = bias_t
        self.result_t = result_t
        self.accum_t = accum_t

    def get_config(self):
        config = {
            'strategy': self.strategy,
            'weight_t': self.weight_t,
            'bias_t': self.bias_t,
            'result_t': self.result_t,
            'accum_t': self.accum_t,
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BatchNormalization(L.BatchNormalization):

    @create_init(L.BatchNormalization)
    def __init__(self, scale_t=None, bias_t=None, result_t=None):
        self.scale_t = scale_t
        self.bias_t = bias_t
        self.result_t = result_t
        assert self.renorm == False
        assert self.virtual_batch_size == None
        assert self.adjustment == None
        self.fused = False

    def get_config(self):
        config = {
            'scale_t': self.scale_t,
            'bias_t': self.bias_t,
            'result_t': self.result_t,
        }
        base_config = super(BatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Activations

class Softmax(L.Softmax):
    def __init__(self, axis=-1, strategy='latency', exp_table_t=None, inv_table_t=None, **kwargs):
        super().__init__(axis=axis, **kwargs)
        self.strategy = strategy
        self.exp_table_t = exp_table_t
        self.inv_table_t = inv_table_t

    def get_config(self):
        config = {
            'strategy': self.strategy,
            'exp_table_t': self.exp_table_t,
            'inv_table_t': self.inv_table_t,
        }
        base_config = super(Softmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ReLU(L.ReLU):
    @create_init(L.ReLU)
    def __init__(self, table_t=None, table_size=None):
        self.table_t = table_t
        self.table_size = table_size

    def get_config(self):
        config = {
            'table_t': self.table_t,
            'table_size': self.table_size,
        }
        base_config = super(ReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class LeakyReLU(L.LeakyReLU):
    @create_init(L.LeakyReLU)
    def __init__(self, table_t=None, table_size=None):
        self.table_t = table_t
        self.table_size = table_size

    def get_config(self):
        config = {
            'table_t': self.table_t,
            'table_size': self.table_size,
        }
        base_config = super(LeakyReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
