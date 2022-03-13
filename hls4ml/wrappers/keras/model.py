
from tensorflow import keras as K

from hls4ml.wrappers.keras.op_wrapper import compile_model

class Model(K.Model):
    pass

class Sequential(K.Sequential):
    def __init__(self, default_precision, layers=None, name=None, output_dir=None):
        super().__init__(layers, name)
        self.default_precision = default_precision
        self.output_dir = output_dir
        # TODO all HLSConfig/Model parameters should be passable to this instance

    def get_config(self):
        config = {
            'default_precision': self.default_precision,
        }
        base_config = super(Sequential, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                steps_per_execution=None,
                **kwargs):

        compile_model(self, output_dir=self.output_dir)

        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            **kwargs
        )

    def rebuild(self, rebuilt_layers):
        new_layers = []
        old_weights = {}
        input_shape = self.input.shape

        while(True):
            if len(self._self_tracked_trackables) > 0:
                old_layer = self._self_tracked_trackables[-1]
                weights = old_layer.get_weights()
                config = old_layer.get_config()
                try:
                    self.pop() # I wish they returned the layer name or some other handle
                except TypeError:
                    break
                if isinstance(old_layer, K.layers.InputLayer):
                    continue
                if old_layer.name in rebuilt_layers:
                    # Create the new wrapped layer instance based on the config
                    new_layer = rebuilt_layers[old_layer.name].from_config(config)
                else:
                    # Or non-wrapped instance iw we skip wrapping
                    new_layer = old_layer.__class__.from_config(config)
                if len(weights) > 0:
                    # Save the weights
                    old_weights[old_layer.name] = weights
                new_layer.keras_class = old_layer.__class__
                new_layers.append(new_layer)
            else:
                break

        for new_layer in reversed(new_layers):
            self.add(new_layer)

        self.build(input_shape)

        # We add weights after the model is built (this is safer than model.set_weights())
        for layer in self.layers:
            weights = old_weights.get(layer.name, None)
            if weights is not None:
                layer.set_weights(weights)

    def strip_wrappers(self):
        new_layers = []
        old_weights = {}
        input_shape = self.input.shape

        while(True):
            if len(self._self_tracked_trackables) > 0:
                old_layer = self._self_tracked_trackables[-1]
                weights = old_layer.get_weights()
                config = old_layer.get_config()
                try:
                    self.pop() # I wish they returned the layer name or some other handle
                except TypeError:
                    break
                if isinstance(old_layer, K.layers.InputLayer):
                    continue
                new_layer = old_layer.keras_class.from_config(config)
                if len(weights) > 0:
                    # Save the weights
                    old_weights[old_layer.name] = weights
                new_layer.hls4ml_class = old_layer.__class__
                new_layers.append(new_layer)
            else:
                break

        for new_layer in reversed(new_layers):
            self.add(new_layer)

        self.build(input_shape)

        # We add weights after the model is built (this is safer than model.set_weights())
        for layer in self.layers:
            weights = old_weights.get(layer.name, None)
            if weights is not None:
                layer.set_weights(weights)

    def get_hls4ml_config(self):
        config = {}

        model_config = {}
        model_config['Precision'] = self.default_precision
        model_config['ReuseFactor'] = 1
        model_config['Strategy'] = 'Latency'

        config['Model'] = model_config

        name_config = {}
        for layer in self.layers:
            try:
                hls_keras_config = layer.get_hls4ml_config()
                if len(hls_keras_config) > 0:
                    hls_config = {}
                    precision_keys = [k for k in hls_keras_config.keys() if k.endswith('_t')]
                    other_keys = [k for k in hls_keras_config.keys() if not k.endswith('_t')]
                    
                    if len(precision_keys) > 0:
                        hls_config['Precision'] = {}
                        for k in precision_keys:
                            hls_param = hls_keras_config[k]
                            if hls_param is not None:
                                precision_name = k.replace('_t', '') if k.endswith('_t') else k #TODO Update when config scheme is overhauled
                                hls_config['Precision'][precision_name] = hls_param
                    
                    if 'skip_wrapping' in other_keys:
                        # Remove 'skip_wrapping' as it is not part of the hls4ml conversion config
                        other_keys.remove('skip_wrapping')
                    for k in other_keys:
                        hls_param = hls_keras_config[k]
                        if hls_param is not None:
                            hls_config[k] = hls_param

                    name_config[layer.name] = hls_config
            except:
                pass

        config['LayerName'] = name_config

        return config
