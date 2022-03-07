#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "op_utils.h"
#include "nnet_utils/nnet_helpers.h"
#include "nnet_utils/nnet_padding.h"
#include "nnet_utils/nnet_padding_stream.h"

using namespace tensorflow;

//hls4ml insert defines

//hls4ml insert parameters

//hls4ml insert typedef-config

//hls4ml insert io-type

#include "io_type.h"

#ifndef NAME
    #define NAME "HZeroPadding2D"
#endif

class HZeroPadding2DOp : public OpKernel {
    public:
        explicit HZeroPadding2DOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensor
            const Tensor& input_tensor = context->input(0);
            auto data = input_tensor.flat<float>().data();

            // Extract the padding information
            const Tensor& padding_tensor = context->input(1);
            auto paddings = padding_tensor.flat<int32>().data();

            const int32 pad_top = paddings[2];
            const int32 pad_bottom = paddings[3];
            const int32 pad_left = paddings[4];
            const int32 pad_right = paddings[5];

            // Create an output tensor
            Tensor* output_tensor = NULL;
            TensorShape out_shape = TensorShape({input_tensor.dim_size(0), pad_top + input_tensor.dim_size(1) + pad_bottom, pad_left + input_tensor.dim_size(2) + pad_right, input_tensor.dim_size(3)});
            OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_tensor));
            auto res = output_tensor->flat<float>().data();

            const int n_batch = input_tensor.dim_size(0);
            const int height = input_tensor.dim_size(1);
            const int width = input_tensor.dim_size(2);

            assert(height == hconfig::in_height && width == hconfig::in_width && "Input tensor size must match the provided config");
            assert(pad_top + height + pad_bottom == hconfig::out_height && pad_left + width + pad_right == hconfig::out_width && "Output tensor size must match the provided config");
            assert(pad_top == hconfig::pad_top && pad_bottom == hconfig::pad_bottom && pad_left == hconfig::pad_left && pad_right == hconfig::pad_right && "Padding pattern must match the provided config");

            ap_data_t<input_t, hconfig::in_height * hconfig::in_width * hconfig::n_chan> ap_data;
            ap_res_t<result_t, hconfig::out_height * hconfig::out_width * hconfig::n_chan> ap_res;

            for (int b = 0; b < n_batch; b++) {
                copy_input<input_t, hconfig::in_height * hconfig::in_width * hconfig::n_chan>(data, ap_data, b);
                nnet::zeropad2d_cl<input_t, result_t, hconfig>(ap_data, ap_res);
                copy_result<result_t, hconfig::out_height * hconfig::out_width * hconfig::n_chan>(ap_res, res, b);
            }
        }
};

REGISTER_KERNEL_BUILDER(Name(NAME).Device(DEVICE_CPU), HZeroPadding2DOp);

REGISTER_OP(NAME)
    .Input("input: float")
    .Input("paddings: int32")
    .Output("result: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        // Paddings is a matrix of [input_rank, 2].
        shape_inference::ShapeHandle paddings;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &paddings));

        // n_dim and input.rank are equivalent.
        shape_inference::ShapeHandle input = c->input(0);
        shape_inference::DimensionHandle n_dim = c->Dim(paddings, 0);
        if (c->ValueKnown(n_dim)) {
            TF_RETURN_IF_ERROR(c->WithRank(input, c->Value(n_dim), &input));
        } else if (c->RankKnown(input)) {
            TF_RETURN_IF_ERROR(c->WithValue(n_dim, c->Rank(input), &n_dim));
        }

        const Tensor* paddings_t = c->input_tensor(1);
        if (paddings_t == nullptr) {
            if (c->ValueKnown(n_dim)) {
                // Make output with n_dim unknown dims.
                c->set_output(0, c->UnknownShapeOfRank(c->Value(n_dim)));
            } else {
                c->set_output(0, c->UnknownShape());
            }
            return Status::OK();
        }

        const int64_t num_dims = paddings_t->shape().dim_size(0);
        TF_RETURN_IF_ERROR(c->WithRank(input, num_dims, &input));
        TF_RETURN_IF_ERROR(c->WithValue(n_dim, num_dims, &n_dim));
        std::vector<shape_inference::DimensionHandle> dims(num_dims);
        auto paddings_data = paddings_t->matrix<int32>();
        for (int64_t i = 0; i < num_dims; ++i) {
            const int32 pad0 = paddings_data(i, 0);
            const int32 pad1 = paddings_data(i, 1);
            TF_RETURN_IF_ERROR(c->Add(c->Dim(input, i), pad0 + pad1, &dims[i]));
        }
        
        c->set_output(0, c->MakeShape(dims));
        return Status::OK();
    });
