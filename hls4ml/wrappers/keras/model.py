
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
        input_shape = self.input.shape
        while(True):
            if len(self._self_tracked_trackables) > 0:
                old_layer = self._self_tracked_trackables[-1]
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
                new_layers.append(new_layer)
            else:
                break

        for new_layer in reversed(new_layers):
            self.add(new_layer)

        self.build(input_shape)
