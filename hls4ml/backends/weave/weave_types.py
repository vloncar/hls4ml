from hls4ml.backends.fpga.fpga_types import (
    ArrayVariableConverter,
    InplaceStreamVariableConverter,
    StreamVariableConverter,
    VariableDefinition,
)

# region ArrayVariable


class WeaveArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        return '{type} {name}{suffix}[{shape}]'.format(
            type=self.type.name, name=self.name, suffix=name_suffix, shape=self.size_cpp()
        )


class WeaveInplaceArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class WeaveArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Weave', definition_cls=WeaveArrayVariableDefinition)


class WeaveInplaceArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Weave', definition_cls=WeaveInplaceArrayVariableDefinition)


# endregion

# region StreamVariable


class WeaveStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        if as_reference:  # Function parameter
            return f'hls::stream<{self.type.name}> &{self.name}{name_suffix}'
        else:  # Declaration
            return 'hls::stream<{type}> {name}{suffix}("{name}")'.format(
                type=self.type.name, name=self.name, suffix=name_suffix
            )


class WeaveInplaceStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class WeaveStreamVariableConverter(StreamVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Weave', definition_cls=WeaveStreamVariableDefinition)


# endregion

# region ScalarStreamVariable
#
# hls4ml's stock StreamVariable wraps the element type in a PackedType covering the whole last
# dimension, i.e. a Dense layer's output stream carries ONE beat holding all n_out values. That is
# precisely what makes stock io_stream Dense batch-in/batch-out and prevents any dot->axpy overlap.
#
# Weave's fused regions need a genuine scalar FIFO -- one element per beat, n_out beats -- so the
# consuming axpy can start on the producer's first output. This converter therefore keeps the plain
# scalar type and only attaches a shallow STREAM depth pragma.


class WeaveScalarStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        if as_reference:  # Function parameter
            return f'hls::stream<{self.type.name}> &{self.name}{name_suffix}'
        else:  # Declaration
            return 'hls::stream<{type}> {name}{suffix}("{name}")'.format(
                type=self.type.name, name=self.name, suffix=name_suffix
            )


class WeaveScalarStreamVariableConverter:
    """Convert a tensor variable into a scalar hls::stream channel (NOT PackedType-wrapped)."""

    def __init__(self, type_converter):
        self.type_converter = type_converter
        self.definition_cls = WeaveScalarStreamVariableDefinition

    def convert(self, tensor_var, depth=8):
        if isinstance(tensor_var, self.definition_cls):  # Already converted
            return tensor_var

        tensor_var.pragma = ('stream', depth)
        tensor_var.type = self.type_converter.convert(tensor_var.type)
        tensor_cls_fqn = tensor_var.__class__.__module__ + '.' + tensor_var.__class__.__qualname__

        tensor_var.__class__ = type(
            'WeaveScalarStreamVariable', (type(tensor_var), self.definition_cls), {'_wrapped': tensor_cls_fqn}
        )
        return tensor_var


# endregion

# region InplaceStreamVariable


class WeaveInplaceStreamVariableConverter(InplaceStreamVariableConverter):
    def __init__(self, type_converter):
        super().__init__(
            type_converter=type_converter, prefix='Weave', definition_cls=WeaveInplaceStreamVariableDefinition
        )


# endregion
