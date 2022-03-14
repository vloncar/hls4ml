#include <omp.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#undef MIN
#undef MAX

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "nnet_utils/nnet_helpers.h"
#include "nnet_utils/nnet_types.h"
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"

using namespace tensorflow;

//hls4ml insert defines

//hls4ml insert parameters

//hls4ml insert typedef-config

//hls4ml insert io-type

#include "io_type.h"

#ifndef NAME
    #define NAME "HActivation"
#endif

#ifndef ACTIVATION
    #define ACTIVATION linear
#endif

#ifndef PARAMETER
    #define PARAMETER 1.0
#endif

class HParametrizedActivationOp : public OpKernel {
    public:
        explicit HParametrizedActivationOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha));
        }

        void Compute(OpKernelContext* context) override {
            // Get the input tensor
            const Tensor& input_tensor = context->input(0);
            auto data = input_tensor.flat<float>().data();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            TensorShape out_shape = TensorShape({input_tensor.dim_size(0), input_tensor.dim_size(1)});
            OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_tensor));
            auto res = output_tensor->flat<float>().data();

            const int n_batch = input_tensor.dim_size(0);
            const int n_in = input_tensor.dim_size(1);

            assert(n_in == hconfig::n_in && "Input tensor size must match the provided config");

            ap_data_t<input_t, hconfig::n_in> ap_data;
            ap_res_t<result_t, hconfig::n_in> ap_res;
            input_t ap_alpha = alpha;

            #pragma omp parallel for private(ap_data, ap_res)
            for (int b = 0; b < n_batch; b++) {
                copy_input<input_t, hconfig::n_in>(data, ap_data, b);
                nnet::ACTIVATION<input_t, result_t, hconfig>(ap_data, ap_alpha, ap_res);
                copy_result<result_t, hconfig::n_in>(ap_res, res, b);
            }

        }

    private:
        float alpha;
};

REGISTER_KERNEL_BUILDER(Name(NAME).Device(DEVICE_CPU), HParametrizedActivationOp);

REGISTER_OP(NAME)
    .Attr("alpha: float") // We don't set the default here
    .Input("input: float")
    .Output("result: float")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);
