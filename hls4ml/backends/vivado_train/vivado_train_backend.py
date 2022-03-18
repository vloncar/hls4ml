import os

from hls4ml.backends import VivadoBackend
from hls4ml.model.flow import get_backend_flows, get_flow, register_flow

class VivadoTrainBackend(VivadoBackend):
    def __init__(self):
        super(VivadoBackend, self).__init__(name='VivadoTrain')
        self._register_flows()
    
    def _register_flows(self):

        templates = self._get_layer_templates()
        template_flow = register_flow('apply_templates', templates, requires=['vivado:init_layers'], backend=self.name)

        writer_passes = [
            'vivadotrain:write_hls'
        ]
        writer_flow_requirements = ['optimize', 'vivado:specific_types', template_flow]
        self._writer_flow = register_flow('write', writer_passes, requires=writer_flow_requirements, backend=self.name)

        required_flows = [
            'vivado:init_layers',
            'vivado:specific_types',
            'vivado:apply_templates'
        ]
        self._default_flow = register_flow('generate_ops', None, requires=required_flows, backend=self.name)
    
    def compile(self, model):
        curr_dir = os.getcwd()
        os.chdir(model.config.get_output_dir())

        lib_name = None
        try:
            ret_val = os.system('bash build_op.sh')
            if ret_val != 0:
                raise Exception('Failed to compile project "{}"'.format(model.config.get_project_name()))
            lib_name = '{}/{}_op-{}.so'.format(model.config.get_output_dir(), model.config.get_project_name(), model.config.get_config_value('Stamp'))
        finally:
            os.chdir(curr_dir)
        
        return lib_name