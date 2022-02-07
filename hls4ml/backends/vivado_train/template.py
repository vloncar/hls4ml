from hls4ml.backends.template import Template

class OpDefinesTemplate(Template):
    def __init__(self, layer_class):
        if isinstance(layer_class, (list, tuple, set)):
            name = '_'.join([cls.__name__.lower() for cls in layer_class])
        else:
            name = layer_class.__name__.lower()
        name += '_op_defines_template'
        super().__init__(name, layer_class, 'op_defines_cpp')