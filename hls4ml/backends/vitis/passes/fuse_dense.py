from copy import copy

import numpy as np

from hls4ml.model.layers import Activation, Dense, HardActivation, ParametrizedActivation, PReLU
from hls4ml.model.optimizer import ModelOptimizerPass, OptimizerPass
from hls4ml.model.types import NamedType

# Activations a Dense kernel can compute on the value it has just produced, grouped by what else the
# kernel needs. softmax is absent by nature: it needs every output of the layer before producing any.

# Computed from the value alone
INLINE_ACTIVATIONS = ('linear', 'relu', 'binary_tanh', 'ternary_tanh')

# Computed from the value alone, by reading a table; the kernel also needs the size of the table
TABLE_ACTIVATIONS = ('sigmoid', 'tanh', 'softplus', 'softsign', 'selu')

# Computed from the value and one number shared by every value
SCALAR_PARAM_ACTIVATIONS = ('leaky_relu', 'thresholded_relu', 'elu')

# Computed from the value and two numbers shared by every value
HARD_ACTIVATIONS = ('hard_sigmoid', 'hard_tanh')

FUSED = 'fused'


def _is_fused(node):
    return isinstance(node, Dense) and str(node.get_attr('strategy', '')).lower() == FUSED


def _graph_class(node):
    """The class of the layer as the graph defines it.

    A backend makes a subclass of every layer class to add its own attributes, so the class of a layer in
    a built model is VitisActivation rather than Activation and comparing types directly never matches.
    """
    for cls in type(node).__mro__:
        if cls.__module__ == Activation.__module__:
            return cls
    return type(node)


def _foldable_activation(node):
    """Return the activation this layer computes, or None if it cannot be folded.

    The classes are compared exactly rather than with isinstance: they all inherit from Activation, and
    treating a ParametrizedActivation or a PReLU as a plain one would drop the numbers it carries.
    """
    cls = _graph_class(node)

    if cls is Activation:
        name = node.get_attr('activation', '').lower()
        return name if name in INLINE_ACTIVATIONS + TABLE_ACTIVATIONS else None

    if cls is HardActivation:
        name = node.get_attr('activation', '').lower()
        return name if name in HARD_ACTIVATIONS else None

    if cls is ParametrizedActivation:
        name = node._get_act_function_name().lower()
        return name if name in SCALAR_PARAM_ACTIVATIONS else None

    # PReLU is foldable in principle, but its numbers are weights of the activation layer and would
    # have to move to the Dense layer. Left for later.
    if cls is PReLU:
        return None

    return None


class FoldActivationIntoFused(OptimizerPass):
    """Compute an activation at the end of the Dense layer before it and remove the separate layer.

    Besides saving a process in the region, this keeps two Dense layers neighbours, without which
    PlanDenseFusion finds no chain. The numbers the kernel needs are copied to the Dense layer.
    """

    def match(self, node):
        if _foldable_activation(node) is None:
            return False
        prev = node.get_input_node()
        if prev is None or not _is_fused(prev):
            return False
        return len(prev.get_output_nodes()) == 1 and prev.get_attr('fused_activation') is None

    def transform(self, model, node):
        prev = node.get_input_node()
        activation = _foldable_activation(node)
        prev.set_attr('fused_activation', activation)

        # The Dense layer takes over the rounding the activation layer did, so the chain carries the
        # same types as it would without the fold. preact_t is what the activation is computed on.
        out_var = prev.get_output_variable()
        prev.set_attr('fused_preact_t', NamedType(f'{prev.name}_preact_t', copy(out_var.type.precision)))
        out_var.type.precision = copy(node.get_output_variable().type.precision)

        if activation in TABLE_ACTIVATIONS:
            if node.get_attr('table_size') is not None:
                prev.set_attr('fused_table_size', node.get_attr('table_size'))
            # Set on the Dense layer, the type is also declared: the types of a layer are the
            # attributes that hold one.
            if node.get_attr('table_t') is not None:
                prev.set_attr('fused_table_t', node.get_attr('table_t'))

        # Each number keeps the type hls4ml gave it; the three are not the same, and rounding one of
        # them to a different type changes the result.
        if activation in SCALAR_PARAM_ACTIVATIONS:
            prev.set_attr('fused_activation_param', node.get_attr('activ_param', 1.0))
            if node.get_attr('param_t') is not None:
                prev.set_attr('fused_param_t', node.get_attr('param_t'))

        if activation in HARD_ACTIVATIONS:
            prev.set_attr('fused_activation_slope', node.get_attr('slope', 0.2))
            prev.set_attr('fused_activation_shift', node.get_attr('shift', 0.5))
            if node.get_attr('slope_t') is not None:
                prev.set_attr('fused_slope_t', node.get_attr('slope_t'))
            if node.get_attr('shift_t') is not None:
                prev.set_attr('fused_shift_t', node.get_attr('shift_t'))

        model.remove_node(node)
        return True


class PlanDenseFusion(ModelOptimizerPass):
    """Group Dense layers into regions and give each layer the form it is computed in.

    A Dense layer needs every input to make an output, so it can pass data one value at a time on one
    side only: ``dot`` reads an array and writes value by value, ``axpy`` reads value by value and
    writes an array. Layers alternate dot, axpy, ... so that every pair overlaps in time; an odd chain
    gets a leading ``plain`` layer so that it still starts and ends on an array, as the model requires.
    """

    def __init__(self):
        pass

    def transform(self, model):
        # io_stream carries a whole row per read, which the fused kernels cannot use. Reported
        # separately; here it only stops the pass.
        if model.config.get_config_value('IOType') != 'io_parallel':
            return False

        changed = False
        fused_layers = []
        for run in self._dense_runs(model):
            forms = self._assign_forms(len(run))
            for layer, form in zip(run, forms):
                if layer.get_attr('fused_form') != form:
                    layer.set_attr('fused_form', form)
                    changed = True
                fused_layers.append(layer)

        if not changed:
            return False

        self._set_parallel_multipliers(fused_layers)
        self._mark_streamed_outputs(fused_layers)

        # The layers of a region run at the same time, so the top function is a DATAFLOW region
        if any(layer.get_attr('fused_form') in ('dot', 'axpy') for layer in fused_layers):
            model.config.pipeline_style = 'dataflow'

        return True

    @staticmethod
    def _assign_forms(length):
        """Return the form of each layer of a chain of the given length."""
        if length < 2:
            return ['plain'] * length
        head = ['plain'] if length % 2 else []
        body = ['dot' if k % 2 == 0 else 'axpy' for k in range(length - len(head))]
        return head + body

    def _dense_runs(self, model):
        """Return the chains of Dense layers that can be fused.

        A chain runs while each layer uses the fused strategy and is the only reader of the one before
        it. Another layer in between, or a second reader, ends it.
        """
        runs, current = [], []
        for layer in model.get_layers():
            if _is_fused(layer) and len(layer.get_output_nodes()) <= 1:
                if current and layer.get_input_node() is not current[-1]:
                    runs.append(current)
                    current = []
                current.append(layer)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        return [run for run in runs if len(run) > 1]

    @staticmethod
    def _lanes_dimension(layer):
        """The number of values the layer works through: n_in for dot, which adds up the inputs, n_out
        for the other forms, which produce the outputs."""
        if layer.get_attr('fused_form') == 'dot':
            return int(layer.get_attr('n_in'))
        return int(layer.get_attr('n_out'))

    def _set_parallel_multipliers(self, layers):
        """Turn the reuse factor of each layer into a count of multipliers used at the same time.

        The two are the same quantity written the other way round, so a given reuse factor asks for the
        same hardware here as with the existing strategies. A dot and axpy pair is given the lower of
        the two counts, since it runs only as fast as its slower half.
        """
        for layer in layers:
            n_in = int(layer.get_attr('n_in'))
            n_out = int(layer.get_attr('n_out'))
            reuse = max(1, int(layer.get_attr('reuse_factor', 1) or 1))
            wanted = max(1, (n_in * n_out) // reuse)
            # More multipliers than values to work through would leave some unused
            layer.set_attr('fused_multipliers', min(wanted, self._lanes_dimension(layer)))

        for layer in layers:
            if layer.get_attr('fused_form') != 'dot':
                continue
            consumers = layer.get_output_nodes()
            if consumers and consumers[0].get_attr('fused_form') == 'axpy':
                pair = (layer, consumers[0])
                shared = min(int(n.get_attr('fused_multipliers')) for n in pair)
                for n in pair:
                    n.set_attr('fused_multipliers', shared)

    @staticmethod
    def _mark_streamed_outputs(layers):
        """Mark the outputs written one value at a time, which TransformTypes turns into streams.

        Only a dot layer read by an axpy layer writes that way.
        """
        for layer in layers:
            consumers = layer.get_output_nodes()
            streams = (
                layer.get_attr('fused_form') == 'dot'
                and len(consumers) == 1
                and consumers[0].get_attr('fused_form') == 'axpy'
            )
            layer.set_attr('fused_stream_out', bool(streams))


class LayoutFusedDotWeights(OptimizerPass):
    """Transpose the weights of a dot layer into the order that kernel reads them.

    hls4ml stores a Dense weight for input i and output j at i * n_out + j, which is what axpy reads.
    dot produces one output at a time and needs the inputs of one output together, at j * n_in + i.
    """

    def match(self, node):
        return (
            isinstance(node, Dense)
            and node.get_attr('fused_form') == 'dot'
            and not node.get_attr('fused_weights_transposed')
        )

    def transform(self, model, node):
        weight = node.weights['weight']
        weight.data = np.ascontiguousarray(weight.data.T)
        weight.shape = list(weight.data.shape)
        node.set_attr('fused_weights_transposed', True)
        return True
