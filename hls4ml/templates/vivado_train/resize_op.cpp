#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "op_utils.h"
#include "nnet_utils/nnet_helpers.h"
#include "nnet_utils/nnet_image.h"
#include "nnet_utils/nnet_image_stream.h"

using namespace tensorflow;

//hls4ml insert defines

//hls4ml insert parameters

//hls4ml insert typedef-config

//hls4ml insert io-type

#include "io_type.h"

#ifndef NAME
    #define NAME "HResize"
#endif

#ifndef ALGORITHM
    #define ALGORITHM resize_nearest
#endif

class HResizeOp : public OpKernel {
    public:
        explicit HResizeOp(OpKernelConstruction* context) : OpKernel(context) {
            std::vector<int32> factor;
            OP_REQUIRES_OK(context, context->GetAttr("factor", &factor));
            factor_height = factor[0];
            factor_width = factor[1];
        }

        void Compute(OpKernelContext* context) override {
            // Grab the input tensor
            const Tensor& input_tensor = context->input(0);
            auto data = input_tensor.flat<float>().data();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            TensorShape out_shape = TensorShape({input_tensor.dim_size(0), factor_height * input_tensor.dim_size(1), factor_width * input_tensor.dim_size(2), input_tensor.dim_size(3)});
            OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_tensor));
            auto res = output_tensor->flat<float>().data();

            const int n_batch = input_tensor.dim_size(0);
            const int height = input_tensor.dim_size(1);
            const int width = input_tensor.dim_size(2);

            assert(height == hconfig::height && width == hconfig::width && "Input tensor size must match the provided config");
            assert(height * factor_height == hconfig::new_height && width * factor_width == hconfig::new_width && "Output tensor size must match the provided config");

            ap_data_t<input_t, hconfig::height * hconfig::width * hconfig::n_chan> ap_data;
            ap_res_t<result_t, hconfig::new_height * hconfig::new_width * hconfig::n_chan> ap_res;

            for (int b = 0; b < n_batch; b++) {
                copy_input<input_t, hconfig::height * hconfig::width * hconfig::n_chan>(data, ap_data, b);
                nnet::ALGORITHM<input_t, hconfig>(ap_data, ap_res);
                copy_result<result_t, hconfig::new_height * hconfig::new_width * hconfig::n_chan>(ap_res, res, b);
            }
        }
    
    private:
        int factor_height;
        int factor_width;
};

REGISTER_KERNEL_BUILDER(Name(NAME).Device(DEVICE_CPU), HResizeOp);

REGISTER_OP(NAME)
    .Attr("factor: list(int) = [2,2]")
    .Input("input: float")
    .Output("result: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        auto batch = c->Dim(c->input(0), 0);
        auto height = c->Dim(c->input(0), 1);
        auto width = c->Dim(c->input(0), 2);
        auto n_chan = c->Dim(c->input(0), 3);
        
        std::vector<int32> factor;
        c->GetAttr("factor", &factor);
        int factor_height = factor[0];
        int factor_width = factor[1];

        auto new_height = c->MakeDim(c->Value(height) * factor_height);
        auto new_width = c->MakeDim(c->Value(width) * factor_width);

        c->set_output(0, c->MakeShape({batch, new_height, new_width, n_chan}));

        return Status::OK();
    });
