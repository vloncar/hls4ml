import os
import tensorflow as tf
import glob
from collections import OrderedDict
from shutil import copyfile, copytree, rmtree

from hls4ml.writer.vivado_writer import VivadoWriter

class VivadoTrainWriter(VivadoWriter):

    def write_project_dir(self, model):
        prj_path = '{}'.format(model.config.get_output_dir())
        if not os.path.isdir(prj_path):
            os.makedirs(prj_path)

    def write_cpp_op(self, model):
        #######################
        ## <layer>_op.cpp
        #######################

        main_layer = list(model.get_layers())[1]

        layer_class = main_layer.class_name.lower() # Assuming first layer is Input, next is our main layer

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, f'../templates/vivado_train/{layer_class}_op.cpp'), 'r')
        fout = open('{}/{}_op.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()), 'w')

        for line in f.readlines():

            if '//hls4ml insert defines' in line:
                newline = ''
                all_precision = OrderedDict()
                for layer in model.get_layers():
                    layer_precision = layer.get_layer_precision()
                    for type_name, type_var in layer_precision.items():
                        # Ensure that layer's types doesn't override existing types
                        # This can happen in case of InplaceVariable types
                        if type_name not in all_precision:
                            all_precision[type_name] = type_var
                for used_type in all_precision.values():
                    newline += used_type.definition_cpp()
                
                for layer in model.get_layers():
                    defines = layer.get_attr('op_defines_cpp', None)
                    if defines is not None:
                        newline += defines
                
                newline += '\n' + '#define NAME "' + model.config.get_project_name() + '"\n'
            
            elif '//hls4ml insert parameters' in line:
                for layer in model.get_layers():
                    config = layer.get_attr('config_cpp', None)
                    if config:
                        newline = '// ' + layer.name + '\n'
                        newline += config

            elif '//hls4ml insert typedef-config' in line:
                # Extract config name
                config_cpp = main_layer.get_attr('config_cpp', None)
                if config_cpp is not None:
                    config_name = config_cpp.split(':')[0].replace('struct', '').strip()
                    newline = f'typedef {config_name} hconfig;\n'
                else:
                    newline = ''
            
            elif '//hls4ml insert io-type' in line:
                newline = '#define IO_TYPE {}'.format(model.config.get_config_value("IOType"))
            
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_build_script(self, model):
        ###################
        # build_op.sh
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir,'../templates/vivado_train/build_op.sh'),'r')
        fout = open('{}/build_op.sh'.format(model.config.get_output_dir()), 'w')

        for line in f.readlines():
            if 'OP_SRC=' in line:
                newline = 'OP_SRC={}_op.cpp\n'.format(model.config.get_project_name())
            elif 'TARGET_LIB=' in line:
                newline = 'TARGET_LIB={}_op.so\n'.format(model.config.get_project_name())
            elif 'TF_CFLAGS=' in line:
                newline = 'TF_CFLAGS="{}"\n'.format(' '.join(tf.sysconfig.get_compile_flags()))
            elif 'TF_LFLAGS=' in line:
                newline = 'TF_LFLAGS="{}"\n'.format(' '.join(tf.sysconfig.get_link_flags()))
            else:
                newline = line
            
            fout.write(newline)

        f.close()
        fout.close()

    def write_nnet_utils(self, model):
        ###################
        ## nnet_utils
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir,'../templates/vivado/nnet_utils/')
        dstpath = '{}/nnet_utils/'.format(model.config.get_output_dir())

        if not os.path.exists(dstpath):
            os.mkdir(dstpath)

        headers = [os.path.basename(h) for h in glob.glob(srcpath + '*.h')]

        for h in headers:
            copyfile(srcpath + h, dstpath + h)

        ###################
        ## ap_types
        ###################

        srcpath = os.path.join(filedir,'../templates/vivado/ap_types/')
        dstpath = '{}/ap_types/'.format(model.config.get_output_dir())

        if os.path.exists(dstpath):
            rmtree(dstpath)

        copytree(srcpath, dstpath)

        ###################
        ## extras
        ###################

        srcpath = os.path.join(filedir,'../templates/vivado_train/op_utils.h')
        dstpath = '{}/op_utils.h'.format(model.config.get_output_dir())
        copyfile(srcpath, dstpath)

        srcpath = os.path.join(filedir,'../templates/vivado_train/io_type.h')
        dstpath = '{}/io_type.h'.format(model.config.get_output_dir())
        copyfile(srcpath, dstpath)

    def write_hls(self, model):
        self.write_project_dir(model)
        self.write_cpp_op(model)
        self.write_build_script(model)
        self.write_nnet_utils(model)
