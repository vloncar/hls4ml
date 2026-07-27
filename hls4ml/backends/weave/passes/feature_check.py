from hls4ml.model.optimizer import ModelOptimizerPass


class ValidateWeaveIoType(ModelOptimizerPass):
    """Weave requires io_parallel.

    Weave fuses adjacent layers into a single DATAFLOW region connected by *scalar* hls::stream FIFOs
    (one element per beat), so a producer's first output can start the consumer immediately. hls4ml's
    io_stream instead wraps a layer's whole output in a PackedType (one beat = the entire feature
    vector), which cannot express that scalar channel. Until IOType becomes a proper per-backend
    config, enforce it defensively here rather than silently mis-generating.
    """

    def __init__(self):
        pass

    def transform(self, model):
        io_type = model.config.get_config_value('IOType')
        if io_type != 'io_parallel':
            raise Exception(
                f'Weave backend requires IOType "io_parallel" (got "{io_type}"). Weave builds its own '
                'scalar-stream fused dataflow regions internally; hls4ml\'s io_stream PackedType channels '
                'are incompatible with that. Use io_parallel, or the Vitis backend for io_stream.'
            )
        return False  # no graph mutation
