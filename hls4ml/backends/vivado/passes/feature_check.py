from hls4ml.model.optimizer import OptimizerPass

# Flow that holds the passes of the fused strategy. A backend that does not register it cannot run it.
FUSION_FLOW = 'fuse_dense'


class ValidateFusedStrategy(OptimizerPass):
    """Stop a build that asks for the fused strategy on a backend that does not provide it.

    The configured strategy is read rather than the layer attribute, so layer types whose initializer
    ignores the value are reported instead of falling back to another strategy silently.
    """

    def match(self, node):
        # Layers that take no strategy at all, such as the input, are not what the setting is about
        if node.get_attr('strategy') is None:
            return False
        if str(node.model.config.get_strategy(node)).lower() != 'fused':
            return False
        backend = node.model.config.backend
        return f'{backend.name.lower()}:{FUSION_FLOW}' not in backend.get_available_flows()

    def transform(self, model, node):
        raise Exception(
            f'Layer "{node.name}" ({node.class_name}) has strategy = "fused", which the '
            f'{model.config.backend.name} backend does not support. Use the Vitis backend, or one of the '
            'strategies this backend provides.'
        )
