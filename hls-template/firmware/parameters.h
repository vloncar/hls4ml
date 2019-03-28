#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#ifndef __INTELFPGA_COMPILER__
#include "ref/ac_int.h"
#include "ref/ac_fixed.h"
#else
#include "HLS/ac_int.h"
#include "HLS/ac_fixed.h"
#endif
#include "nnet_utils/nnet_dense.h"
//#include "nnet_utils/nnet_conv.h"
//#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_activation.h"
//#include "nnet_utils/nnet_common.h"
//#include "nnet_utils/nnet_batchnorm.h"
//#include "nnet_utils/nnet_pooling.h"
//#include "nnet_utils/nnet_merge.h"

//hls-fpga-machine-learning insert numbers

//hls-fpga-machine-learning insert layer-precision

//hls-fpga-machine-learning insert layer-config

#endif 
