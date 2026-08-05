from hls4ml.backends.fpga.fpga_types import APTypeConverter, HLSTypeConverter
from hls4ml.backends.vitis.vitis_types import VitisScalarStreamVariableConverter
from hls4ml.model.optimizer import GlobalOptimizerPass
from hls4ml.model.types import InplaceTensorVariable


class TransformFusedTypes(GlobalOptimizerPass):
    """Convert the connections inside a fused region into scalar streams.

    The 'fused' strategy places a chain of Dense layers into a single DATAFLOW region. There, a layer
    can start computing as soon as the previous layer produces its first output value, instead of
    waiting for the complete output array. This requires the connection between the two layers to
    carry one value at a time, so the array variable is replaced by a scalar ``hls::stream``.

    All other variables keep the array type assigned by ``vivado:transform_types``. This pass therefore
    runs after that one and changes only the variables the fusion planner has marked.

    Only connections between two layers of the same region are converted. The input of the first layer
    and the output of the last layer stay arrays, so the region has the same interface as a single
    layer and can be used in a model whose remaining layers are not fused.

    The fusion planner marks the connections by setting attributes on the producing node:

        fused_stream_out (bool): the output of this node is a connection inside a fused region.
        fused_stream_depth (int, optional): stream depth. The converter default is used if not set.
    """

    def __init__(self):
        self.type_converter = HLSTypeConverter(precision_converter=APTypeConverter())
        self.scalar_stream_var_converter = VitisScalarStreamVariableConverter(type_converter=self.type_converter)

    def transform(self, model, node):
        if not node.get_attr('fused_stream_out', False):
            return False

        depth = int(node.get_attr('fused_stream_depth', 0) or 0)

        transformed = False
        for out_name, var in node.variables.items():
            # The output of the model is the output of the region and stays an array.
            if out_name in node.model.outputs:
                continue
            # An in-place variable refers to its input instead of holding its own data, so there is
            # nothing to convert here. The node that writes the data is converted instead.
            if isinstance(var, InplaceTensorVariable):
                continue

            new_var = self.scalar_stream_var_converter.convert(var, depth=depth)
            node.set_attr(out_name, new_var)
            transformed = True

        return transformed
