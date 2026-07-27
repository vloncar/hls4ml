#ifndef NNET_DENSE_OP_H_
#define NNET_DENSE_OP_H_

#include "nnet_common.h"
#include "nnet_function_stubs.h"
#include "nnet_mult.h"

// Outer-product resource Dense kernel, parameterized by reuse_factor (drop-in alternative to
// dense_resource). hls4ml-native weight layout w[i_in*n_out + i_out], clean contiguous addressing:
//   block_factor BF = n_in*n_out/rf  MACs/cycle
//   P_OUT = min(BF, n_out) output lanes ; P_IN = BF/P_OUT input-reduction lanes ; cycles = rf
// Valid rf (hls4ml rule) guarantees P_OUT | n_out, P_IN | n_in, P_OUT*P_IN = BF.
namespace nnet {

// DenseKernel wrapper so the config `kernel` typedef can select it (reads sizes from CONFIG_T).
template <class data_T, class res_T, typename CONFIG_T> class DenseOp : public DenseKernel<data_T, res_T, CONFIG_T> {
  public:
    static void dense(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                      typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                      typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
        #pragma HLS INLINE
        const int N_IN = CONFIG_T::n_in, N_OUT = CONFIG_T::n_out, RF = CONFIG_T::reuse_factor;
        const int BF = (N_IN * N_OUT) / RF;
        const int P_OUT = (BF <= N_OUT) ? BF : N_OUT;
        const int P_IN = BF / P_OUT;

        typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
        #pragma HLS ARRAY_PARTITION variable=acc complete
        #pragma HLS ARRAY_RESHAPE variable=weights cyclic factor=BF dim=1

        for (int j = 0; j < N_OUT; j++) {
            #pragma HLS UNROLL
            acc[j] = (typename CONFIG_T::accum_t)biases[j];
        }
        for (int ii = 0; ii < N_IN; ii += P_IN) {
            for (int io = 0; io < N_OUT; io += P_OUT) {
                #pragma HLS PIPELINE II=1
                for (int po = 0; po < P_OUT; po++) {
                    #pragma HLS UNROLL
                    typename CONFIG_T::accum_t s = 0;
                    for (int pi = 0; pi < P_IN; pi++) {
                        #pragma HLS UNROLL
                        s += static_cast<typename CONFIG_T::accum_t>(
                            CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(
                                data[ii + pi], weights[(ii + pi) * N_OUT + (io + po)]));
                    }
                    acc[io + po] += s;
                }
            }
        }
        for (int j = 0; j < N_OUT; j++) {
            #pragma HLS UNROLL
            res[j] = cast<data_T, res_T, CONFIG_T>(acc[j]);
        }
    }
};

} // namespace nnet

#endif
