"""Weave layer fusion (BLAISE_PLAN 2.3e).

A Dense layer is all-to-all, so it can stream on exactly one side. Its two duals are:

    dot  : array in  -> stream out   (batch-in,  stream-out)
    axpy : stream in -> array out    (stream-in, batch-out)

A boundary overlaps iff the producer streams out AND the consumer streams in, so a run of Dense
layers is assigned alternating dot/axpy forms: every dot->axpy boundary overlaps inside one DATAFLOW
region, while each axpy->dot boundary is a plain array handoff (a natural barrier, no wasted buffer).
Measured effect: 2.7-3.6x lower end-to-end latency vs the same kernels run sequentially.

Region boundary rule: the model input and output are arrays (io_parallel top level), so a run must
START by consuming an array (dot or plain) and END by producing an array (axpy or plain). Working
backwards from an axpy tail, an even-length run is dot/axpy/...; an odd-length run gets a leading
'plain' layer so the remainder stays even.
"""

import numpy as np

from hls4ml.model.layers import Activation, Dense
from hls4ml.model.optimizer import ModelOptimizerPass, OptimizerPass

# Elementwise activations Weave can fold into a Dense kernel's output stage. relu/linear are inline;
# tanh/sigmoid use a lookup ROM built inside the kernel (hls4ml's exact tables). Reductions (softmax)
# and parametric activations (elu/prelu/leaky) are NOT foldable and stay separate layers.
FOLDABLE_ACTIVATIONS = ('relu', 'linear', 'tanh', 'sigmoid')


class WeaveFoldActivation(OptimizerPass):
    """Fold an elementwise ReLU/linear activation into the producing Dense kernel.

    Removes the Activation layer entirely: it would otherwise sit between two Dense layers as its own
    dataflow process, costing an extra FIFO and breaking Dense-run adjacency for the planner.
    """

    def match(self, node):
        if not isinstance(node, Activation):
            return False
        if node.get_attr('activation', '').lower() not in FOLDABLE_ACTIVATIONS:
            return False
        prev = node.get_input_node()
        return (
            isinstance(prev, Dense)
            and len(prev.get_output_nodes()) == 1
            and prev.get_attr('weave_act') is None
        )

    # NOTE: folding runs before plan_fusion so the Dense layers it joins up become directly adjacent
    # and the planner sees one long run instead of several length-1 runs.

    def transform(self, model, node):
        prev = node.get_input_node()
        act = node.get_attr('activation', 'linear').lower()
        prev.set_attr('weave_act', act)
        # carry the table SIZE for table-based activations (plain int, safe). table_t stays the default
        # ap_fixed<18,8> literal in the template -- the activation's named typedef vanishes with the
        # removed layer, and a custom table precision is a rare case handled later if needed.
        if act in ('tanh', 'sigmoid') and node.get_attr('table_size') is not None:
            prev.set_attr('weave_table_size', node.get_attr('table_size'))
        model.remove_node(node)
        return True


class WeavePlanFusion(ModelOptimizerPass):
    """Assign alternating dot/axpy forms to maximal linear runs of Dense layers."""

    def __init__(self):
        pass  # ModelOptimizerPass.__init__ takes (name, transform); subclasses override it

    def transform(self, model):
        # Fusion targets the io_parallel top level: the model input/output are arrays, matching the
        # dot(array-in) / axpy(array-out) region boundaries. io_stream's PackedType channels cannot
        # express a scalar dot->axpy FIFO.
        if model.config.get_config_value('IOType') != 'io_parallel':
            return False

        changed = False
        for run in self._dense_runs(model):
            # respect hand-pinned runs: if any layer in the run was pinned, the user is driving
            if any(layer.get_attr('weave_form', 'auto') != 'auto' for layer in run):
                continue
            for layer, form in zip(run, self._assign_forms(len(run))):
                if layer.get_attr('weave_form') != form:
                    layer.set_attr('weave_form', form)
                    changed = True

        if changed:
            self._fix_par_entries(model)
        return changed

    @staticmethod
    def _assign_forms(length):
        """Forms for a run of `length` Dense layers; must start array-in and end array-out."""
        if length < 2:
            return ['plain'] * length
        head = ['plain'] if length % 2 else []
        body = ['dot' if k % 2 == 0 else 'axpy' for k in range(length - len(head))]
        return head + body

    def _dense_runs(self, model):
        """Maximal runs of directly-adjacent, single-consumer Dense layers."""
        runs, current = [], []
        for layer in model.get_layers():
            fusible = isinstance(layer, Dense) and len(layer.get_output_nodes()) <= 1
            if fusible:
                # break the run if this Dense is not fed directly by the previous one
                if current and layer.get_input_node() is not current[-1]:
                    runs.append(current)
                    current = []
                current.append(layer)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        return [r for r in runs if r]

    @staticmethod
    def _stride_dim(layer):
        """The dimension a form's MAC lanes stride over: dot reduces over inputs (`for i += PAR`) so it
        strides n_in; axpy/plain scatter across outputs (`for jb += PAR`) so they stride n_out."""
        n_in = int(layer.get_attr('n_in'))
        n_out = int(layer.get_attr('n_out'))
        return n_in if layer.get_attr('weave_form') == 'dot' else n_out

    def _fix_par_entries(self, model):
        """Cap par_entries and equalize it across an overlapping dot->axpy pair.

        par_entries need NOT divide the stride dimension: the kernels bounds-check every lane
        (`if (i+p < n_in)` / `if (j < n_out)`) and use cyclic array partitioning, both of which handle a
        non-dividing factor. Forcing divisibility (the old behaviour) collapsed par to 1-2 for
        odd/coprime/non-square shapes and crippled parallelism. So here we only (a) cap par at the
        smallest stride dim in the group -- more lanes than elements is pure waste -- and (b) equalize
        across a dot->axpy pair, since the pair overlaps only as fast as its slower side.
        """
        for layer in model.get_layers():
            if layer.get_attr('weave_form') not in ('plain', 'dot', 'axpy'):
                continue

            group = [layer]
            consumers = layer.get_output_nodes()
            if layer.get_attr('weave_form') == 'dot' and consumers and consumers[0].get_attr('weave_form') == 'axpy':
                group.append(consumers[0])
            elif layer.get_attr('weave_form') == 'axpy':
                continue  # already handled as its producer's pair

            requested = min(int(n.get_attr('par_entries', 1) or 1) for n in group)
            par = max(1, min(requested, min(self._stride_dim(n) for n in group)))

            for n in group:
                n.set_attr('par_entries', par)


class WeaveLayoutDotWeights(OptimizerPass):
    """Emit dot-form weights output-major: w[j * n_in + i].

    hls4ml stores a Dense kernel as (n_in, n_out), whose C-order flatten is w[i * n_out + j] --
    already exactly what the axpy form wants. Only dot layers need the transpose.
    """

    def match(self, node):
        return (
            isinstance(node, Dense)
            and node.get_attr('weave_form') == 'dot'
            and not node.get_attr('_weave_dot_layout')
        )

    def transform(self, model, node):
        weight = node.weights['weight']
        weight.data = np.ascontiguousarray(weight.data.T)
        weight.shape = list(weight.data.shape)
        node.set_attr('_weave_dot_layout', True)
        return True
