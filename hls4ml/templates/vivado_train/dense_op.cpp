#include <omp.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "op_utils.h"
#include "nnet_utils/nnet_helpers.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_stream.h"

using namespace tensorflow;

//hls4ml insert defines

//hls4ml insert parameters

//hls4ml insert typedef-config

//hls4ml insert io-type

#include "io_type.h"

#ifndef NAME
    #define NAME "HDense"
#endif

class HDenseOp : public OpKernel {
    public:
        explicit HDenseOp(OpKernelConstruction* context) : OpKernel(context) {}

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
            TensorShape out_shape = TensorShape({input_tensor.dim_size(0), bias_tensor.dim_size(0)});
            OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_tensor));
            auto res = output_tensor->flat<float>().data();

            const int n_batch = input_tensor.dim_size(0);
            const int n_in = input_tensor.dim_size(1);
            const int n_out = weight_tensor.dim_size(1);

            assert(n_in == hconfig::n_in && "Input tensor size must match the provided config");
            assert(n_out == hconfig::n_out && "Output tensor size must match the provided config");

            ap_data_t<input_t, hconfig::n_in> ap_data;
            ap_res_t<result_t, hconfig::n_out> ap_res;
            typename hconfig::weight_t ap_weights[n_in * n_out];
            typename hconfig::bias_t ap_biases[n_out];

            #pragma omp parallel for
            for(int w = 0; w < n_in * n_out; w++) {
                ap_weights[w] = (typename hconfig::weight_t) weights(w);
            }

            #pragma omp parallel for
            for(int b = 0; b < n_out; b++) {
                ap_biases[b] = (typename hconfig::bias_t) biases(b);
            }

            #pragma omp parallel for private(ap_data, ap_res)
            for (int b = 0; b < n_batch; b++) {
                copy_input<input_t, hconfig::n_in>(data, ap_data, b);
                nnet::dense<input_t, result_t, hconfig>(ap_data, ap_res, ap_weights, ap_biases);
                copy_result<result_t, hconfig::n_out>(ap_res, res, b);
            }

        }
};

REGISTER_KERNEL_BUILDER(Name(NAME).Device(DEVICE_CPU), HDenseOp);

REGISTER_OP(NAME)
    .Input("input: float")
    .Input("weights: float")
    .Input("bias: float")
    .Output("result: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        auto batch = c->Dim(c->input(0), 0);
        auto n_in = c->Dim(c->input(0), 1);
        auto n_out = c->Dim(c->input(1), 1);

        c->set_output(0, c->MakeShape({batch, n_out}));

        return Status::OK();
    });