from hls4ml.backends.fpga.fpga_types import VariableDefinition

# region ScalarStreamVariable

# Used by the 'fused' strategy for the connections between the layers of a fused region.


class VitisScalarStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        if as_reference:  # Function parameter
            return f'hls::stream<{self.type.name}> &{self.name}{name_suffix}'
        else:  # Declaration
            return 'hls::stream<{type}> {name}{suffix}("{name}")'.format(
                type=self.type.name, name=self.name, suffix=name_suffix
            )


class VitisScalarStreamVariableConverter:
    """Convert a tensor variable into a scalar ``hls::stream``.

    ``StreamVariableConverter`` wraps the type in a ``PackedType`` that covers the last dimension, so a
    single read of the stream returns all values of that dimension. This converter does not, so a read
    returns one value and the variable keeps its original precision.
    """

    def __init__(self, type_converter):
        self.type_converter = type_converter
        self.definition_cls = VitisScalarStreamVariableDefinition

    def convert(self, tensor_var, depth=0):
        if isinstance(tensor_var, self.definition_cls):  # Already converted
            return tensor_var

        if depth == 0:
            depth = 8  # small, because the stream passes values on rather than storing the tensor
        tensor_var.pragma = ('stream', depth)
        tensor_var.type = self.type_converter.convert(tensor_var.type)

        tensor_cls_fqn = tensor_var.__class__.__module__ + '.' + tensor_var.__class__.__qualname__

        # The definition class is placed first, unlike in the converters in fpga_types. Those run while
        # the variable is still a plain TensorVariable, whereas this conversion happens after
        # TransformTypes has made it an ArrayVariable. In the other order Python would find the array
        # definition_cpp first and declare the variable as an array, while still applying the stream
        # pragma to it.
        tensor_var.__class__ = type(
            'VitisScalarStreamVariable', (self.definition_cls, type(tensor_var)), {'_wrapped': tensor_cls_fqn}
        )

        return tensor_var


# endregion
