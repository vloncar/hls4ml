from hls4ml.model.layers import Activation, BatchNormalization, Dense, PReLU, ParametrizedActivation, Softmax
from hls4ml.backends.vivado_train.template import OpDefinesTemplate

# Dense templates

activation_op_defines_template = """#define ACTIVATION {activation}\n"""

class ActivationDefinesTemplate(OpDefinesTemplate):
    def __init__(self):
        super().__init__(Activation)
        self.template = activation_op_defines_template
    
    def format(self, node):
        return self.template.format(activation=node.get_attr('activation'))


# TODO implement parametrized activation templates