import os
import sys
import subprocess
from pathlib import Path
from warnings import warn

from hls4ml.backends import FPGABackend
from hls4ml.model.attributes import ConfigurableAttribute
from hls4ml.model.flow import register_flow
from hls4ml.model.layers import Dense, Layer
from hls4ml.model.optimizer import layer_optimizer
from hls4ml.report import parse_vivado_report


class WeaveBackend(FPGABackend):
    """A standalone hls4ml backend whose Dense/Conv matmul cores are built on the Vitis BLAS L1
    primitives (plus a fixed-point-aware compute epilogue). Inherits from ``FPGABackend`` (reusing its
    compile harness, reuse-factor math, precision conversion, im2col codegen) but NOT from the
    Vivado/Vitis backends, so it pulls in none of their passes via auto-discovery. See BLAISE_PLAN.md."""

    def __init__(self):
        super().__init__('Weave')
        self._register_layer_attributes()
        self._register_flows()

    def _register_layer_attributes(self):
        # Weave-specific parallelism knob for the BLAS matmul cores (MAC lanes = 2**par_entries_log2)
        attrs = self.attribute_map.get(Dense, [])
        attrs.append(ConfigurableAttribute('par_entries', default=1, description='Weave BLAS parallel MAC lanes'))
        # Fusion form: 'auto' lets the weave:plan_fusion planner choose, but a run can be pinned by
        # hand to 'plain' | 'dot' | 'axpy'. See backends/weave/passes/fuse_dense.py.
        attrs.append(
            ConfigurableAttribute('weave_form', value_type=str, default='auto', description='Fusion form')
        )
        attrs.append(
            ConfigurableAttribute('weave_fifo_depth', default=8, description='dot->axpy channel FIFO depth')
        )
        self.attribute_map[Dense] = attrs

    def _register_flows(self):
        initializers = self._get_layer_initializers()
        init_flow = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        # Streaming graph fixups reused from backends/fpga/passes (auto-discovered under weave:)
        streaming_passes = [
            'weave:clone_output',
        ]
        streaming_flow = register_flow('streaming', streaming_passes, requires=[init_flow], backend=self.name)

        optimization_passes = [
            # fail fast if the model is not io_parallel (Weave's fused scalar-stream regions require it)
            'weave:validate_weave_io_type',
            'infer_precision_types',
            # Weave layer fusion. Order matters: fold elementwise activations into the Dense kernels
            # first so Dense layers become directly adjacent, then plan the dot/axpy alternation over
            # the resulting runs, then lay out each dot layer's weights output-major.
            # (BatchNorm folding is already handled upstream by the generic 'convert' flow.)
            'weave:weave_fold_activation',
            'weave:weave_plan_fusion',
            'weave:weave_layout_dot_weights',
        ]
        optimization_flow = register_flow('optimize', optimization_passes, requires=[init_flow], backend=self.name)

        weave_types = [
            'weave:transform_types',
            'weave:set_pipeline_style',
        ]
        # requires the optimize flow: transform_types/set_pipeline_style both read the fusion form
        # that plan_fusion assigns, so the planner must have run first.
        weave_types_flow = register_flow('specific_types', weave_types, requires=[optimization_flow], backend=self.name)

        template_flow = register_flow('apply_templates', self._get_layer_templates, requires=[init_flow], backend=self.name)

        writer_passes = ['make_stamp', 'weave:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['weave:ip'], backend=self.name)

        # NOTE: Weave is intentionally minimal (Dense + activations). Passes auto-discovered from
        # backends/fpga/passes that aren't wired into a flow here are simply unused; they get wired in
        # as the corresponding features (conv, pooling, etc.) are added. No warning is emitted for them.

        ip_flow_requirements = [
            'optimize',
            init_flow,
            streaming_flow,
            optimization_flow,
            weave_types_flow,
            template_flow,
        ]
        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def get_default_flow(self):
        return self._default_flow

    def get_writer_flow(self):
        return self._writer_flow

    def create_initial_config(
        self,
        part='xcu250-figd2104-2L-e',
        clock_period=5,
        clock_uncertainty='27%',
        io_type='io_stream',
        namespace=None,
        write_weights_txt=True,
        write_tar=False,
        **_,
    ):
        """Create the initial configuration of the Weave backend.

        Args:
            part (str, optional): The FPGA part to be used. Defaults to 'xcu250-figd2104-2L-e'.
            clock_period (int, optional): The clock period. Defaults to 5.
            clock_uncertainty (str, optional): The clock uncertainty. Defaults to 27%.
            io_type (str, optional): Type of implementation used ('io_parallel' or 'io_stream').
                Defaults to 'io_stream'.
            namespace (str, optional): If defined, place all generated code within a namespace.
            write_weights_txt (bool, optional): If True, writes weights to .txt files (faster compile).
            write_tar (bool, optional): If True, compresses the output directory into a .tar.gz file.

        Returns:
            dict: initial configuration.
        """
        config = {}
        config['Part'] = part if part is not None else 'xcu250-figd2104-2L-e'
        config['ClockPeriod'] = clock_period if clock_period is not None else 5
        config['ClockUncertainty'] = clock_uncertainty if clock_uncertainty is not None else '27%'
        config['IOType'] = io_type if io_type is not None else 'io_stream'
        config['HLSConfig'] = {}
        config['WriterConfig'] = {
            'Namespace': namespace,
            'WriteWeightsTxt': write_weights_txt,
            'WriteTar': write_tar,
            'TBOutputStream': 'both',
            'WriteEmulationConstants': False,
        }
        return config

    def build(
        self,
        model,
        reset=False,
        csim=True,
        synth=True,
        cosim=False,
        validation=False,
        export=False,
        vsynth=False,
        fifo_opt=False,
        log_to_stdout=True,
    ):
        if 'linux' in sys.platform:
            found_vrun = os.system('command -v vitis-run > /dev/null') == 0
            if not found_vrun:
                raise Exception('Vitis installation not found. Make sure "vitis-run" is on PATH.')

        build_opts = (
            'array set opt {\n'
            f'    reset      {int(reset)}\n'
            f'    csim       {int(csim)}\n'
            f'    synth      {int(synth)}\n'
            f'    cosim      {int(cosim)}\n'
            f'    validation {int(validation)}\n'
            f'    export     {int(export)}\n'
            f'    vsynth     {int(vsynth)}\n'
            f'    fifo_opt   {int(fifo_opt)}\n'
            '}\n'
        )

        tcl_path = Path(model.config.get_output_dir()) / 'build_opt.tcl'
        with open(tcl_path, 'w') as file:
            file.write(build_opts)

        output_dir = model.config.get_output_dir()
        build_command = 'vitis-run --tcl build_prj.tcl --mode hls'
        stdout_log = os.path.join(output_dir, 'build_stdout.log')
        stderr_log = os.path.join(output_dir, 'build_stderr.log')
        stdout_target = None if log_to_stdout else open(stdout_log, 'w')
        stderr_target = None if log_to_stdout else open(stderr_log, 'w')
        try:
            process = subprocess.Popen(
                build_command, shell=True, cwd=output_dir, stdout=stdout_target, stderr=stderr_target, text=True
            )
            process.communicate()
            if process.returncode != 0:
                raise Exception(f'Build failed for {model.config.get_project_name()}. See logs for details.')
        finally:
            if not log_to_stdout:
                stdout_target.close()
                stderr_target.close()

        return parse_vivado_report(output_dir)

    # ---- layer initializers ----

    @layer_optimizer(Layer)
    def init_base_layer(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('reuse_factor', reuse_factor)

        target_cycles = layer.model.config.get_target_cycles(layer)
        layer.set_attr('target_cycles', target_cycles)

    @layer_optimizer(Dense)
    def init_dense(self, layer):
        # Weave always uses its own BLAS-style kernel, which manages parallelism via `par_entries`
        # (MAC lanes across outputs). We tag the layer 'resource' so the io_stream dense wrapper does
        # NOT force-pipeline (which would fully unroll the kernel); the kernel does its own pipelining.
        layer.set_attr('strategy', 'resource')

        # par_entries need NOT divide n_out: the kernels bounds-check every lane and use cyclic array
        # partitioning, both of which tolerate a non-dividing factor. So we only clamp to [1, max dim] --
        # more lanes than elements is pure waste. Form-aware capping/equalizing happens later in the
        # weave:weave_plan_fusion pass (which also knows dot strides n_in vs axpy/plain strides n_out).
        n_in, n_out = self.get_layer_mult_size(layer)
        par = int(layer.get_attr('par_entries', 1) or 1)
        par = max(1, min(par, max(n_in, n_out)))
        layer.set_attr('par_entries', par)
