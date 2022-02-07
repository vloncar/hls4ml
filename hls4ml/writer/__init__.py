from __future__ import absolute_import

from hls4ml.writer.writers import Writer, register_writer, get_writer
from hls4ml.writer.vivado_writer import VivadoWriter
from hls4ml.writer.vivado_accelerator_writer import VivadoAcceleratorWriter
from hls4ml.writer.vivado_train_writer import VivadoTrainWriter
from hls4ml.writer.quartus_writer import QuartusWriter

register_writer('Vivado', VivadoWriter)
register_writer('VivadoAccelerator', VivadoAcceleratorWriter)
register_writer('VivadoTrain', VivadoTrainWriter)
register_writer('Quartus', QuartusWriter)
