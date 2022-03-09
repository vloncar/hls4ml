#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "op_utils.h"
#include "nnet_utils/nnet_helpers.h"
#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_conv2d_stream.h"

using namespace tensorflow;

//hls4ml insert defines

//hls4ml insert parameters

//hls4ml insert typedef-config

//hls4ml insert io-type

#include "io_type.h"

#ifndef NAME
    #define NAME "HConv2D"
#endif

class HConv2DOp : public OpKernel {
    public:
        explicit HConv2DOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            // Get the input tensors (input, weights & bias)
            const Tensor& input_tensor = context->input(0);
            const Tensor& weight_tensor = context->input(1);
            const Tensor& bias_tensor = context->input(2);
            auto data = input_tensor.flat<float>().data();
            auto weights = weight_tensor.flat<float>();
            auto biases = bias_tensor.flat<float>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            TensorShape out_shape = TensorShape({input_tensor.dim_size(0), hconfig::out_height, hconfig::out_width, hconfig::n_filt});
            OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_tensor));
            auto res = output_tensor->flat<float>().data();

            const int n_batch = input_tensor.dim_size(0);
            const int height = input_tensor.dim_size(1);
            const int width = input_tensor.dim_size(2);

            assert(height == hconfig::in_height && width == hconfig::in_width && "Input tensor size must match the provided config");
            //TODO Add a check for output size

            ap_data_t<input_t, hconfig::in_height * hconfig::in_width * hconfig::n_chan> ap_data;
            ap_res_t<result_t, hconfig::out_height * hconfig::out_width * hconfig::n_filt> ap_res;
            typename hconfig::weight_t ap_weights[hconfig::filt_height * hconfig::filt_width * hconfig::n_chan * hconfig::n_filt];
            typename hconfig::bias_t ap_biases[hconfig::n_filt];

            CopyWeights: for(int w = 0; w < hconfig::filt_height * hconfig::filt_width * hconfig::n_chan * hconfig::n_filt; w++) {
                ap_weights[w] = (typename hconfig::weight_t) weights(w);
            }

            CopyBiases: for(unsigned b = 0; b < hconfig::n_filt; b++) {
                ap_biases[b] = (typename hconfig::bias_t) biases(b);
            }

            for (int b = 0; b < n_batch; b++) {
                copy_input<input_t, hconfig::in_height * hconfig::in_width * hconfig::n_chan>(data, ap_data, b);
                nnet::conv_2d_cl<input_t, result_t, hconfig>(ap_data, ap_res, ap_weights, ap_biases);
                copy_result<result_t, hconfig::out_height * hconfig::out_width * hconfig::n_filt>(ap_res, res, b);
            }
        }
};

REGISTER_KERNEL_BUILDER(Name(NAME).Device(DEVICE_CPU), HConv2DOp);

REGISTER_OP(NAME)
    .Input("input: float")
    .Input("filter: float")
    .Input("bias: float")
    .Output("output: float")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true") // Not used by the implementation, here just for compatibility
    .Attr("padding: {'SAME', 'VALID'}")
    .Attr("explicit_paddings: list(int) = []") // Not used by the implementation, here just for compatibility
    .Attr("data_format: {'NHWC', 'NCHW'} = 'NHWC'")
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn(shape_inference::Conv2DShape);
