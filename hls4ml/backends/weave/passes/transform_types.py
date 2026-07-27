from hls4ml.backends.fpga.fpga_types import APTypeConverter, HLSTypeConverter, StaticWeightVariableConverter
from hls4ml.backends.weave.weave_types import (
    WeaveArrayVariableConverter,
    WeaveInplaceArrayVariableConverter,
    WeaveInplaceStreamVariableConverter,
    WeaveScalarStreamVariableConverter,
    WeaveStreamVariableConverter,
)
from hls4ml.model.optimizer import GlobalOptimizerPass
from hls4ml.model.types import InplaceTensorVariable


class TransformTypes(GlobalOptimizerPass):
    def __init__(self):
        self.type_converter = HLSTypeConverter(precision_converter=APTypeConverter())
        self.array_var_converter = WeaveArrayVariableConverter(type_converter=self.type_converter)
        self.inplace_array_var_converter = WeaveInplaceArrayVariableConverter(type_converter=self.type_converter)
        self.stream_var_converter = WeaveStreamVariableConverter(type_converter=self.type_converter)
        self.scalar_stream_var_converter = WeaveScalarStreamVariableConverter(type_converter=self.type_converter)
        self.inplace_stream_var_converter = WeaveInplaceStreamVariableConverter(type_converter=self.type_converter)
        self.weight_var_converter = StaticWeightVariableConverter(type_converter=self.type_converter)

    def transform(self, model, node):
        io_type = node.model.config.get_config_value('IOType')
        # A fused 'dot' layer streams its outputs one at a time into the consuming axpy; everything
        # else keeps the stock behaviour for the configured io_type.
        is_dot = node.get_attr('weave_form') == 'dot'

        for out_name, var in node.variables.items():
            if is_dot and out_name not in node.model.outputs:
                new_var = self.scalar_stream_var_converter.convert(
                    var, depth=int(node.get_attr('weave_fifo_depth', 8))
                )
            elif io_type == 'io_stream':
                if isinstance(var, InplaceTensorVariable):
                    new_var = self.inplace_stream_var_converter.convert(var)
                else:
                    new_var = self.stream_var_converter.convert(var)
            elif io_type == 'io_serial':
                new_var = self.array_var_converter.convert(var, pragma='stream')
            elif io_type == 'io_parallel':
                if out_name in node.model.inputs:
                    # NOTE this needs to be changed to partition
                    new_var = self.array_var_converter.convert(var, pragma='reshape')
                elif isinstance(var, InplaceTensorVariable):
                    new_var = self.inplace_array_var_converter.convert(var, pragma='')
                else:
                    new_var = self.array_var_converter.convert(var, pragma='partition')
            else:
                raise Exception(f'Unknown IOType {io_type} in {node.name} ({node.__class__.__name__})')

            node.set_attr(out_name, new_var)

        for w_name, weight in node.weights.items():
            new_weight = self.weight_var_converter.convert(weight)
            node.set_attr(w_name, new_weight)

        for t_name, type in node.types.items():
            new_type = self.type_converter.convert(type)
            node.set_attr(t_name, new_type)
