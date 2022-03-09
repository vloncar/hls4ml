#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "op_utils.h"
#include "nnet_utils/nnet_helpers.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"

using namespace tensorflow;

//hls4ml insert defines

//hls4ml insert parameters

//hls4ml insert typedef-config

//hls4ml insert io-type

#include "io_type.h"

#ifndef NAME
    #define NAME "HPooling2D"
#endif

class HPooling2DOp : public OpKernel {
    public:
        explicit HPooling2DOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            // Get the input tensor
            const Tensor& input_tensor = context->input(0);
            auto data = input_tensor.flat<float>().data();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            TensorShape out_shape = TensorShape({input_tensor.dim_size(0), hconfig::out_height, hconfig::out_width, input_tensor.dim_size(3)});
            OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_tensor));
            auto res = output_tensor->flat<float>().data();

            const int n_batch = input_tensor.dim_size(0);
            const int height = input_tensor.dim_size(1);
            const int width = input_tensor.dim_size(2);

            assert(height == hconfig::in_height && width == hconfig::in_width && "Input tensor size must match the provided config");
            //TODO Add a check for output size
            
            ap_data_t<input_t, hconfig::in_height * hconfig::in_width * hconfig::n_chan> ap_data;
            ap_res_t<result_t, hconfig::out_height * hconfig::out_width * hconfig::n_chan> ap_res;

            for (int b = 0; b < n_batch; b++) {
                copy_input<input_t, hconfig::in_height * hconfig::in_width * hconfig::n_chan>(data, ap_data, b);
                nnet::pooling2d_cl<input_t, result_t, hconfig>(ap_data, ap_res);
                copy_result<result_t, hconfig::out_height * hconfig::out_width * hconfig::n_chan>(ap_res, res, b);
            }
        }
};

REGISTER_KERNEL_BUILDER(Name(NAME).Device(DEVICE_CPU), HPooling2DOp);

REGISTER_OP(NAME)
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr("padding: {'SAME', 'VALID'}")
    .Attr("explicit_paddings: list(int) = []") // Not used by the implementation, here just for compatibility
    .Attr("data_format: {'NHWC', 'NCHW'} = 'NHWC'")
    .Input("input: float")
    .Output("output: float")
    .SetShapeFn(shape_inference::MaxPoolShape);
