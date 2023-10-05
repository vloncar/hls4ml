
#ifndef NNET_GATHER_H
#define NNET_GATHER_H

#include "hls_stream.h"
#include "nnet_common.h"

namespace nnet {

struct getitem_config {
    static const unsigned n_in = 20;
    static const unsigned n_out = 10;
    static const unsigned item_index = 1;
};

template <class data_T, typename CONFIG_T> void getitem(data_T input[CONFIG_T::n_in], data_T output[CONFIG_T::n_out]) {

    #pragma HLS PIPELINE

    for (int i = 0; i < CONFIG_T::n_out; i++) {
        output[i] = input[CONFIG_T::item_index * CONFIG_T::n_out + i];
    }
}

struct gather_config {
    static const unsigned n_in = 20;
    static const unsigned n_indices = 10;
};

template <class data_T, class index_T, typename CONFIG_T>
void gather(data_T input[CONFIG_T::n_in], index_T index[CONFIG_T::n_indices], data_T output[CONFIG_T::n_indices]) {

    #pragma HLS PIPELINE

    for (int i = 0; i < CONFIG_T::n_indices; i++) {
        output[i] = input[index[i]];
    }
}

} // namespace nnet

#endif
