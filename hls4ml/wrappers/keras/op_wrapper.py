import json
import tempfile
import re

from tensorflow.python.framework import load_library

from hls4ml.model import ModelGraph
from hls4ml.converters.keras_to_hls import KerasModelReader, layer_handlers
from hls4ml.utils.config import create_config
from hls4ml.wrappers.keras.gradients import register_gradient
from hls4ml.wrappers.keras.calls import lookup_call_wrapper

class CustomOpWrapper(object):
    def __init__(self, keras_layer, hls_model, act_model=None):
        self.keras_layer = keras_layer
        self.hls_model = hls_model
        self.act_model = act_model

        self.name = self.keras_layer.name
        self.op_keras_class = self.keras_layer.__class__.__name__
        self.op_func_name = self._get_python_func_name(self.hls_model.config.get_project_name()) # HDense1 -> h_dense1
        self.op_lib_name = None
        self.op_func = None

        if self.act_model is not None:
            self.act_func_name = self._get_python_func_name(self.act_model.config.get_project_name())
            act = list(self.act_model.get_layers())[1].get_attr('activation')
            self.act_keras_class = ''.join(word.title() for word in act.split('_'))
        else:
            self.act_func_name = None
            self.act_keras_class = None
        self.act_lib_name = None
        self.act_func = None

    def write(self):
        self.hls_model.write()
        if self.act_model is not None:
            self.act_model.write()

    def compile(self):
        self.write()
        # Call backend compile directly, since we're not linking the op
        self.op_lib_name = self.hls_model.config.backend.compile(self.hls_model)
        if self.act_model is not None:
            self.act_lib_name = self.act_model.config.backend.compile(self.act_model)

    def link(self):
        if self.op_lib_name is None:
            raise Exception('Custom op must be compiled first with `compile()`.')

        op_module = load_library.load_op_library(self.op_lib_name)
        self.op_func = op_module.__dict__[self.op_func_name]

        if self.act_lib_name is not None:
            op_module = load_library.load_op_library(self.act_lib_name)
            self.act_func = op_module.__dict__[self.act_func_name]

    def _get_python_func_name(self, cpp_name):
        # Convert CamelCase to snake_case, where single capital letters aren't followed by an _
        # as this seems to be the format that TF uses. For example, LeakyReLU -> leaky_re_lu
        # instead of leaky_re_l_u
        subbed = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', cpp_name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', subbed).lower()

    def _make_init_func(self):
        def init_func(new_self, *args, **kwargs):
            super(new_self.__class__, new_self).__init__(*args, **kwargs)
            new_self.op_func = self.op_func
            new_self.act_func = self.act_func

        return init_func

    def build_layer(self):
        # Note to future self: It may seem more intuitive to just patch the call() function
        # of the original layer, but this will not work. The layers may get built as they are added
        # (e.g., in Sequential model) and TF's AutoGraph won't like it when the ops are switched.
        # Unfortunately, there is no easy way to force a rebuild of the model, the related internal
        # functions make an effort not to rebuild. Instead we create wrappers around existing layer
        # classes with new call functions and replace the original layers in the model with the wrappers.

        register_gradient(self.hls_model.config.get_project_name(), self.op_keras_class)
        if self.act_model is not None:
            register_gradient(self.act_model.config.get_project_name(), self.act_keras_class)

        h_layer = type(
            self.hls_model.config.get_project_name(), # Class name, e.g., HDense1
            (self.keras_layer.__class__,), # Base class, e.g., Dense
            {
                'keras_name': self.name,
                '__init__': self._make_init_func(),
                'call': lookup_call_wrapper(self.op_keras_class),
            }
        )

        return h_layer


def _create_input_layer(input_shape, name):
    input_layer = {}
    input_layer['name'] = name
    input_layer['class_name'] = 'InputLayer'
    input_layer['input_shape'] = input_shape if input_shape[0] is not None else input_shape[1:]

    return input_layer

def _extract_config_from_layer(layer_list, model_config):
    config = {}

    for layer in layer_list:
        hls4ml_config = layer.get('hls4ml_config', {})
        if len(hls4ml_config) > 0:
            precision_keys = [k for k in hls4ml_config.keys() if k.endswith('_t')]
            other_keys = [k for k in hls4ml_config.keys() if not k.endswith('_t')]
            config['Precision'] = {}
            for k in precision_keys:
                value = hls4ml_config[k]
                if value is not None:
                    precision_name = k.replace('_t', '') if k.endswith('_t') else k #TODO Update when config scheme is overhauled
                    config['Precision'][precision_name] = value
            for k in other_keys:
                config[k] = hls4ml_config[k]

    return config

def _parse_model(keras_model, output_dir=None):
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    model_arch = json.loads(keras_model.to_json())
    reader = KerasModelReader(keras_model)

    #Define layers to skip for conversion to HLS
    skip_layers = ['Dropout']

    #Map inputs of skipped and split (activation) layers
    inputs_map = {}

    #Loop through layers
    layer_counter = 0

    input_layers = None
    output_layers = None

    model_config = {
        'Precision': model_arch['config']['default_precision'],
        'ReuseFactor': 1,
    }

    layer_config = None
    if model_arch['class_name'] == 'Sequential':
        layer_config = model_arch['config']
        if 'layers' in layer_config: # Newer Keras versions have 'layers' in 'config' key
            layer_config = layer_config['layers']
    elif model_arch['class_name'] in ['Model', 'Functional']:
        layer_config = model_arch['config']['layers']
        input_layers = [ inp[0] for inp in model_arch['config']['input_layers'] ]
        output_layers = [ out[0] for out in model_arch['config']['output_layers'] ]
    else:
        raise Exception('Unable to parse model class: {}'.format(model_arch['class_name']))

    output_shapes = {}
    output_shape = None

    hls_model_counter = 1
    converted_models = {}

    for keras_layer in layer_config:

        if 'batch_input_shape' in keras_layer['config']:
            if 'inbound_nodes' in keras_layer and len(keras_layer['inbound_nodes']) > 0:
                input_shapes = [output_shapes[inbound_node[0]] for inbound_node in keras_layer['inbound_nodes'][0]]
            else:
                input_shapes = [keras_layer['config']['batch_input_shape']]
        else:
            if 'inbound_nodes' in keras_layer:
                input_shapes = [output_shapes[inbound_node[0]] for inbound_node in keras_layer['inbound_nodes'][0]]
            else:
                # Sequential model, so output_shape from the previous layer is still valid
                input_shapes = [output_shape]

        keras_name = keras_layer['config']['name']
        keras_class = keras_layer['class_name']

        if keras_class in skip_layers:
            if 'inbound_nodes' in keras_layer:
                #Currently supported skipped layers have only one input
                parent_input = keras_layer['inbound_nodes'][0][0][0]
                #Skipped layers can follow each other (e.g., Dropout -> Flatten)
                inputs_map[keras_name] = inputs_map.get(parent_input, parent_input)

            output_shapes[keras_name] = input_shapes[0]

            continue

        layer_counter = layer_counter + 1

        #Extract inbound nodes
        if 'inbound_nodes' in keras_layer and len(keras_layer['inbound_nodes']) > 0:
            input_names = [ inputs_map.get(inp[0], inp[0]) for inp in keras_layer['inbound_nodes'][0] ]
        else:
            input_names = None

        layer, output_shape = layer_handlers[keras_class](keras_layer, input_names, input_shapes, reader, {})
        output_shapes[keras_name] = output_shape

        assert layer['name'] == keras_name
        if layer['class_name'] != 'Activation':
            assert layer['class_name'] == keras_class

        if keras_class != 'InputLayer' and len(layer_list) == 0:
            input_layer = _create_input_layer(input_shapes[0], 'input' + str(layer_counter))
            layer_list.append(input_layer)

        #print('Layer name: {}, layer type: {}, input shapes: {}, output shape: {}'.format(keras_name, keras_class, input_shapes, output_shape))
        layer_list.append(layer)

        layer_hls4ml_config = _extract_config_from_layer(layer_list, model_config)

        hls_model = None
        act_model = None
        if not (len(layer_list) == 1 and layer_list[0]['class_name'] == 'InputLayer'):
            config = create_config(
                output_dir=output_dir + '/' + keras_name,
                project_name='H' + keras_class + str(hls_model_counter),
                backend='VivadoTrain',
                io_type='io_parallel'
            )

            config['HLSConfig'] = {}
            config['HLSConfig']['Model'] = model_config
            config['HLSConfig']['LayerName'] = {
                keras_name: layer_hls4ml_config
            }

            hls_model = ModelGraph(config, reader, layer_list)
            hls_model_counter += 1

            # Reset the layer list
            layer_list = []

        if 'activation' in layer and keras_class not in ['Activation', 'LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU', 'Softmax', 'TernaryTanh'] and layer['activation'] != 'linear':
            input_layer = _create_input_layer(output_shape, 'input' + str(layer_counter))
            layer_list.append(input_layer)
            act_layer = {}
            act_layer['name'] = keras_name + '_' + layer['activation']
            act_layer['activation'] = layer['activation']
            if 'activ_param' in layer:
                act_layer['activ_param'] = layer['activ_param']
                act_layer['class_name'] = layer['activation']
            elif layer['activation'] == 'softmax':
                act_layer['class_name'] = 'Softmax'
                act_layer['axis'] = -1
            else:
                act_layer['class_name'] = 'Activation'
            inputs_map[keras_name] = act_layer['name']
            if output_layers is not None and layer['name'] in output_layers:
                output_layers = [act_layer['name'] if name == keras_name else name for name in output_layers]
            layer_list.append(act_layer)

            config = create_config(
                output_dir='hls4mlprj_train/' + act_layer['name'],
                project_name='H' + act_layer['class_name'] + str(hls_model_counter),
                backend='VivadoTrain',
                io_type='io_parallel'
            )

            config['HLSConfig'] = {}
            config['HLSConfig']['Model'] = model_config
            config['HLSConfig']['LayerName'] = {
                act_layer['name']: layer_hls4ml_config
            }

            act_model = ModelGraph(config, reader, layer_list)
            hls_model_counter += 1

            layer_list = []

        if hls_model is not None:
            original_layer = keras_model.get_layer(keras_name)
            op_wrapper = CustomOpWrapper(original_layer, hls_model, act_model)
            converted_models[keras_name] = [ op_wrapper ]

    return converted_models

def _compile_ops(op_wrappers):
    for name, wrapper_list in op_wrappers.items():
        for wrapper in wrapper_list:
            wrapper.compile()
            wrapper.link()

def _build_layers(op_wrappers):
    new_layers = {}
    for name, wrapper_list in op_wrappers.items():
        for wrapper in wrapper_list:
            h_layer = wrapper.build_layer()
            new_layers[name] = h_layer

    return new_layers

def _rebuild_model(model, rebuilt_layers):
    return model.rebuild(rebuilt_layers)

def compile_model(model, output_dir=None):
    op_wrappers = _parse_model(model, output_dir=output_dir)
    _compile_ops(op_wrappers)
    new_layers = _build_layers(op_wrappers)
    return _rebuild_model(model, new_layers)
