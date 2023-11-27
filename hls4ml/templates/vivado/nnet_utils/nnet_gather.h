
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

struct gather_config_1d {
    static const unsigned n_in = 20;
    static const unsigned n_indices = 10;
};

template <class data_T, class index_T, typename CONFIG_T>
void gather_1d(data_T input[CONFIG_T::n_in], index_T index[CONFIG_T::n_indices], data_T output[CONFIG_T::n_indices]) {

    #pragma HLS PIPELINE

    for (int i = 0; i < CONFIG_T::n_indices; i++) {
        output[i] = input[index[i]];
    }
}

struct gather_2d_config {
    static const unsigned n_in_0 = 20;
    static const unsigned n_in_1 = 20;
    static const unsigned n_indices = 10;
};

template <class data_T, class index_T, typename CONFIG_T>
void gather_2d(data_T input[CONFIG_T::n_in_0 * CONFIG_T::n_in_1], index_T index[CONFIG_T::n_indices],
               data_T output[CONFIG_T::n_indices * CONFIG_T::n_in_1]) {

    #pragma HLS PIPELINE

    for (int i = 0; i < CONFIG_T::n_indices; i++) {
        for (int j = 0; j < CONFIG_T::n_in_1; j++) {
            output[i * CONFIG_T::n_in_1 + j] = input[index[i] * CONFIG_T::n_in_1 + j];
        }
    }
}

struct gather_3d_config {
    static const unsigned n_in_0 = 10;
    static const unsigned n_in_1 = 10;
    static const unsigned n_in_2 = 10;
    static const unsigned n_indices = 10;
};

template <class data_T, class index_T, typename CONFIG_T>
void gather_3d(data_T input[CONFIG_T::n_in_0 * CONFIG_T::n_in_1 * CONFIG_T::n_in_2], index_T index[CONFIG_T::n_indices],
               data_T output[CONFIG_T::n_indices * CONFIG_T::n_in_1 * CONFIG_T::n_in_2]) {

#pragma HLS PIPELINE

    for (int i = 0; i < CONFIG_T::n_indices; i++) {
        for (int j = 0; j < CONFIG_T::n_in_1; j++) {
            for (int k = 0; k < CONFIG_T::n_in_2; k++) {
                output[i * CONFIG_T::n_in_1 * CONFIG_T::n_in_2 + j * CONFIG_T::n_in_2 + k] = input[index[i] * CONFIG_T::n_in_1 * CONFIG_T::n_in_2 + j * CONFIG_T::n_in_2 + k];
            }
        }
    }
}

} // namespace nnet

#endif
