from hls4ml.backends.backend import get_backend
from hls4ml.backends.oneapi.oneapi_template import StreamFunctionCallTemplate, TaskSequenceTemplate
from hls4ml.backends.oneapi.passes.core_templates import (
    batchnorm_config_template,
    batchnorm_function_template,
    batchnorm_include_list,
    batchnorm_stream_function_template,
    batchnorm_task_sequence_template,
)
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.optimizer.passes.qkeras import ApplyAlpha


class ApplyAlphaConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(ApplyAlpha)
        self.template = batchnorm_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['n_in'] = node.get_input_variable().size_cpp()
        params['product_type'] = get_backend('oneAPI').product_type(
            node.get_input_variable().type.precision, node.get_weights('scale').type.precision
        )

        return self.template.format(**params)


class ApplyAlphaFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(ApplyAlpha, include_header=batchnorm_include_list)
        self.template = batchnorm_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['scale'] = node.get_weights('scale').name
        params['bias'] = node.get_weights('bias').name

        return self.template.format(**params)


class ApplyAlphaTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(ApplyAlpha)
        self.template = batchnorm_task_sequence_template

    def format(self, node):
        params = self._default_function_params(node)

        return self.template.format(**params)


class ApplyAlphaStreamFunctionTemplate(StreamFunctionCallTemplate):
    def __init__(self):
        super().__init__(ApplyAlpha)
        self.template = batchnorm_stream_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['scale'] = node.get_weights('scale').name
        params['bias'] = node.get_weights('bias').name

        return self.template.format(**params)
