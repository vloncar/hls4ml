#ifndef NNET_DENSE_WEAVE_H_
#define NNET_DENSE_WEAVE_H_

#include "hls_stream.h"
#include "nnet_activation.h" // reuse hls4ml's exact init_tanh_table / init_sigmoid_table
#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_function_stubs.h"
#include "nnet_mult.h"

// Weave Dense core.
//
// Outer-product / "axpy" formulation of y = W^T x + b, matching hls4ml's native weight layout
// weights[i * n_out + j] (input i, output j). Because that layout is contiguous in the output
// index j, the natural parallelization is across outputs: CONFIG_T::par_entries MAC lanes update
// par_entries accumulators per cycle, reducing over the inputs sequentially. This yields
// ~1 MAC / DSP / cycle (n_in * n_out / par_entries cycles) and keeps ap_fixed quantization exact
// -- the fixed-point tree/lane-reduction lever validated in BLAISE_PLAN.md Phase 0b/2.3e.
//
// Requirements: n_out % par_entries == 0 (checked in the Weave config pass).

namespace nnet {

// Elementwise activation folded into a Dense kernel's output stage (CONFIG_T::activation), so the
// activation needs no separate dataflow process/FIFO. Set by the weave_fold_activation pass.
// Table-based activations (tanh/sigmoid) look up a ROM built with hls4ml's exact table math, so the
// folded result is numerically identical to the standalone activation layer. relu/linear need no table.
enum WeaveAct { WEAVE_LINEAR = 0, WEAVE_RELU = 1, WEAVE_TANH = 2, WEAVE_SIGMOID = 3 };

// Build the lookup ROM for the layer's activation (no-op for relu/linear). Reuses hls4ml's own inits
// so tanh/sigmoid tables match the standalone activation bit-for-bit.
template <typename CONFIG_T> void weave_init_act_table(typename CONFIG_T::table_t table[CONFIG_T::table_size]) {
    if (CONFIG_T::activation == WEAVE_TANH) {
        init_tanh_table<CONFIG_T, CONFIG_T::table_size>(table);
    } else if (CONFIG_T::activation == WEAVE_SIGMOID) {
        init_sigmoid_table<CONFIG_T, CONFIG_T::table_size>(table);
    }
}

// Apply the folded activation to one element. `table` is unused (and DCE'd) for relu/linear.
template <class res_T, typename CONFIG_T>
inline res_T weave_activate(res_T v, const typename CONFIG_T::table_t table[CONFIG_T::table_size]) {
    #pragma HLS INLINE
    if (CONFIG_T::activation == WEAVE_RELU) {
        return v < res_T(0) ? res_T(0) : v;
    } else if (CONFIG_T::activation == WEAVE_TANH) {
        // index math identical to nnet::tanh: domain [-4, 4)
        int index = (int)(v * CONFIG_T::table_size / 8) + CONFIG_T::table_size / 2;
        if (index < 0)
            index = 0;
        if (index > (int)CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        return (res_T)table[index];
    } else if (CONFIG_T::activation == WEAVE_SIGMOID) {
        // index math identical to nnet::sigmoid: domain [-8, 8)
        int index = (int)(v * CONFIG_T::table_size / 16) + CONFIG_T::table_size / 2;
        if (index < 0)
            index = 0;
        if (index > (int)CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        return (res_T)table[index];
    }
    return v; // WEAVE_LINEAR
}

// Declares + fills the folded-activation ROM once at kernel scope. For relu/linear the array is never
// read and is dead-code-eliminated. Used by all three Weave kernels before their output stage.
#define WEAVE_ACT_TABLE(name)                                                                                          \
    typename CONFIG_T::table_t name[CONFIG_T::table_size];                                                            \
    weave_init_act_table<CONFIG_T>(name);

template <class data_T, class res_T, typename CONFIG_T>
class WeaveDense : public DenseKernel<data_T, res_T, CONFIG_T> {
  public:
    static void dense(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                      typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                      typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
        const unsigned PAR = CONFIG_T::par_entries;

        typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
        #pragma HLS ARRAY_PARTITION variable=acc cyclic factor=PAR
        #pragma HLS ARRAY_RESHAPE variable=weights cyclic factor=PAR dim=1
        WEAVE_ACT_TABLE(act_table)

    WeaveInit:
        for (unsigned j = 0; j < CONFIG_T::n_out; j++) {
            #pragma HLS UNROLL factor=PAR
            acc[j] = (typename CONFIG_T::accum_t)biases[j];
        }

    WeaveAccum:
        for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
            data_T cache = data[i];
        WeaveLanes:
            for (unsigned jb = 0; jb < CONFIG_T::n_out; jb += PAR) {
                #pragma HLS PIPELINE II=1
                for (unsigned p = 0; p < PAR; p++) {
                    #pragma HLS UNROLL
                    unsigned j = jb + p;
                    if (j < CONFIG_T::n_out) {
                        acc[j] += CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(
                            cache, weights[i * CONFIG_T::n_out + j]);
                    }
                }
            }
        }

    WeaveResult:
        for (unsigned j = 0; j < CONFIG_T::n_out; j++) {
            #pragma HLS UNROLL factor=PAR
            res[j] = weave_activate<res_T, CONFIG_T>(cast<data_T, res_T, CONFIG_T>(acc[j]), act_table);
        }
    }
};

// ---------------------------------------------------------------------------
// Fused-region kernels (BLAISE_PLAN 2.3e).
//
// A Dense layer is all-to-all, so it can never be stream-in AND stream-out: output j needs every
// input. But it has two duals, and each streams on ONE side:
//
//   weave_dot  : array in  -> stream out   (needs the whole input; emits output j as it finishes)
//   weave_axpy : stream in -> array out    (consumes input i as it arrives; all outputs finalise together)
//
// Alternating them (dot -> axpy) lets the axpy start on the dot's FIRST output instead of its last,
// so the two layers overlap inside one DATAFLOW region. The dot->axpy boundary is a scalar FIFO;
// the axpy->dot boundary is a plain array (a natural barrier, no wasted buffering).
//
// Weight layouts differ and are emitted per-form by the Weave passes/writer:
//   dot  expects w[j * n_in + i]   (output-major; weave_layout_dot_weights transposes for this)
//   axpy expects w[i * n_out + j]  (input-major = hls4ml's native layout, no transpose)
//
// CONFIG_T::act_relu folds an elementwise ReLU into the kernel's output stage.
// ---------------------------------------------------------------------------

// DOT form: array in, stream out. weights[j * n_in + i].
template <class data_T, class res_T, typename CONFIG_T>
void weave_dot(data_T data[CONFIG_T::n_in], hls::stream<res_T> &res,
                typename CONFIG_T::weight_t weights[CONFIG_T::n_out * CONFIG_T::n_in],
                typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    const unsigned PAR = CONFIG_T::par_entries;
    #pragma HLS ARRAY_RESHAPE variable=weights cyclic factor=PAR dim=1
    #pragma HLS ARRAY_RESHAPE variable=data cyclic factor=PAR dim=1
    WEAVE_ACT_TABLE(act_table)

WeaveDotOut:
    for (unsigned j = 0; j < CONFIG_T::n_out; j++) {
        typename CONFIG_T::accum_t part[PAR];
        #pragma HLS ARRAY_PARTITION variable=part complete

    WeaveDotClear:
        for (unsigned p = 0; p < PAR; p++) {
            #pragma HLS UNROLL
            part[p] = (typename CONFIG_T::accum_t)0;
        }

    WeaveDotAccum:
        for (unsigned i = 0; i < CONFIG_T::n_in; i += PAR) {
            #pragma HLS PIPELINE II=1
            for (unsigned p = 0; p < PAR; p++) {
                #pragma HLS UNROLL
                if (i + p < CONFIG_T::n_in) {
                    part[p] += CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(
                        data[i + p], weights[j * CONFIG_T::n_in + i + p]);
                }
            }
        }

        typename CONFIG_T::accum_t acc = (typename CONFIG_T::accum_t)biases[j];
    WeaveDotDrain:
        for (unsigned p = 0; p < PAR; p++) {
            #pragma HLS UNROLL
            acc += part[p];
        }
        // emit output j immediately -- this is what the consuming axpy overlaps with
        res.write(weave_activate<res_T, CONFIG_T>(cast<data_T, res_T, CONFIG_T>(acc), act_table));
    }
}

// AXPY form: stream in, array out. weights[i * n_out + j] (hls4ml-native layout).
template <class data_T, class res_T, typename CONFIG_T>
void weave_axpy(hls::stream<data_T> &data, res_T res[CONFIG_T::n_out],
                 typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                 typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    const unsigned PAR = CONFIG_T::par_entries;

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc cyclic factor=PAR
    #pragma HLS ARRAY_RESHAPE variable=weights cyclic factor=PAR dim=1
    WEAVE_ACT_TABLE(act_table)

WeaveAxpyInit:
    for (unsigned j = 0; j < CONFIG_T::n_out; j++) {
        #pragma HLS UNROLL factor=PAR
        acc[j] = (typename CONFIG_T::accum_t)biases[j];
    }

WeaveAxpyAccum:
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        data_T cache = data.read(); // consume input i as soon as the producer emits it
    WeaveAxpyLanes:
        for (unsigned jb = 0; jb < CONFIG_T::n_out; jb += PAR) {
            #pragma HLS PIPELINE II=1
            for (unsigned p = 0; p < PAR; p++) {
                #pragma HLS UNROLL
                unsigned j = jb + p;
                if (j < CONFIG_T::n_out) {
                    acc[j] += CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(
                        cache, weights[i * CONFIG_T::n_out + j]);
                }
            }
        }
    }

WeaveAxpyResult:
    for (unsigned j = 0; j < CONFIG_T::n_out; j++) {
        #pragma HLS UNROLL factor=PAR
        res[j] = weave_activate<res_T, CONFIG_T>(cast<data_T, res_T, CONFIG_T>(acc[j]), act_table);
    }
}

} // namespace nnet

#endif
