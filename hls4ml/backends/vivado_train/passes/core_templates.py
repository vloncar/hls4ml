from hls4ml.model.layers import Activation, BatchNormalization, Dense, PReLU, ParametrizedActivation, Softmax
from hls4ml.backends.vivado_train.template import OpDefinesTemplate

activation_op_defines_template = """#define ACTIVATION {activation}\n"""

class ActivationDefinesTemplate(OpDefinesTemplate):
    def __init__(self):
        super().__init__(Activation)
        self.template = activation_op_defines_template
    
    def format(self, node):
        return self.template.format(activation=node.get_attr('activation'))


act_parameter_op_defines_template = """#define PARAMETER {parameter}\n"""

class ParametrizedActivationDefinesTemplate(OpDefinesTemplate):
    def __init__(self):
        super().__init__(ParametrizedActivation)
        self.act_template = activation_op_defines_template
        self.param_template = act_parameter_op_defines_template
    
    def format(self, node):
        act_define = self.act_template.format(activation=node._get_act_function_name())
        param_define = self.param_template.format(parameter=node.get_attr('activ_param'))
        return act_define + param_define