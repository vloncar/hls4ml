#include <cmath>
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
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"

using namespace tensorflow;

//hls4ml insert defines

//hls4ml insert parameters

//hls4ml insert typedef-config

//hls4ml insert io-type

#include "io_type.h"

#ifndef NAME
    #define NAME "HBatchNormalization"
#endif

class HBatchNormalizationOp : public OpKernel {
    public:
        explicit HBatchNormalizationOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
        }

        void Compute(OpKernelContext* context) override {
            // Get the input tensors (input, gamma, beta, mean, variance)
            const Tensor& input_tensor = context->input(0);
            const Tensor& gamma_tensor = context->input(1);
            const Tensor& beta_tensor = context->input(2);
            const Tensor& mean_tensor = context->input(3);
            const Tensor& variance_tensor = context->input(4);

            auto data = input_tensor.flat<float>().data();
            auto gamma = gamma_tensor.flat<float>();
            auto beta = beta_tensor.flat<float>();
            auto mean = mean_tensor.flat<float>();
            auto variance = variance_tensor.flat<float>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
            auto res = output_tensor->flat<float>().data();

            // Temporary scale and bias arrays
            Tensor scale_tensor;  // scale = gamma / sqrt(variance + epsilon)
            Tensor bias_tensor; // bias = beta - gamma * mean / sqrt(variance + epsilon)
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value, mean_tensor.shape(), &scale_tensor));
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value, mean_tensor.shape(), &bias_tensor));
            auto scale = scale_tensor.flat<float>();
            auto bias = bias_tensor.flat<float>();

            #pragma omp parallel for
            for (int i = 0; i < scale_tensor.NumElements(); i++) {
                scale(i) = gamma(i) / sqrt(variance(i) + epsilon);
                bias(i) = beta(i) - gamma(i) * mean(i) / sqrt(variance(i) + epsilon);
            }

            const int n_batch = input_tensor.dim_size(0);
            const unsigned n_in = (hconfig::n_filt == -1) ? hconfig::n_in : hconfig::n_filt;

            ap_data_t<input_t, hconfig::n_in> ap_data;
            ap_res_t<result_t, hconfig::n_in> ap_res;
            typename hconfig::scale_t ap_scale[n_in];
            typename hconfig::bias_t ap_bias[n_in];

            #pragma omp parallel for
            for(int s = 0; s < n_in; s++) {
                ap_scale[s] = (typename hconfig::scale_t) scale(s);
            }

            #pragma omp parallel for
            for(int b = 0; b < n_in; b++) {
                ap_bias[b] = (typename hconfig::bias_t) bias(b);
            }

            #pragma omp parallel for private(ap_data, ap_res)
            for (int b = 0; b < n_batch; b++) {
                copy_input<input_t, hconfig::n_in>(data, ap_data, b);
                nnet::normalize<input_t, result_t, hconfig>(ap_data, ap_res, ap_scale, ap_bias);
                copy_result<result_t, hconfig::n_in>(ap_res, res, b);
            }

        }

    private:
        float epsilon;
};

REGISTER_KERNEL_BUILDER(Name(NAME).Device(DEVICE_CPU), HBatchNormalizationOp);

REGISTER_OP(NAME)
    .Attr("epsilon: float = 1e-3")
    .Attr("is_training: bool") // Not really used
    .Input("input: float")
    .Input("scale: float")
    .Input("offset: float")
    .Input("mean: float")
    .Input("variance: float")
    .Output("result: float")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

