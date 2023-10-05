#ifndef NNET_ARRAY_H_
#define NNET_ARRAY_H_

#include <math.h>

namespace nnet {

struct transpose_config {
    static const unsigned height = 10;
    static const unsigned width = 10;
    static const unsigned depth = 10;
    static constexpr unsigned perm[3] = {2, 0, 1};
};

template <class data_T, class res_T, typename CONFIG_T>
void transpose_2d(data_T data[CONFIG_T::height * CONFIG_T::width], res_T data_t[CONFIG_T::height * CONFIG_T::width]) {
    #pragma HLS PIPELINE

    for (int i = 0; i < CONFIG_T::height; i++) {
        for (int j = 0; j < CONFIG_T::width; j++) {
            data_t[j * CONFIG_T::height + i] = data[i * CONFIG_T::width + j];
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void transpose_3d(data_T data[CONFIG_T::depth * CONFIG_T::height * CONFIG_T::width],
                  res_T data_t[CONFIG_T::depth * CONFIG_T::height * CONFIG_T::width]) {
    unsigned dims[3] = {CONFIG_T::depth, CONFIG_T::height, CONFIG_T::width};
    unsigned dims_t[3];
    dims_t[0] = dims[CONFIG_T::perm[0]];
    dims_t[1] = dims[CONFIG_T::perm[1]];
    dims_t[2] = dims[CONFIG_T::perm[2]];

    int idx[3] = {0}, idx_t[3] = {0};
    for (idx[0] = 0; idx[0] < dims[0]; idx[0]++) {
        for (idx[1] = 0; idx[1] < dims[1]; idx[1]++) {
            for (idx[2] = 0; idx[2] < dims[2]; idx[2]++) {
                idx_t[0] = idx[CONFIG_T::perm[0]];
                idx_t[1] = idx[CONFIG_T::perm[1]];
                idx_t[2] = idx[CONFIG_T::perm[2]];

                data_t[idx_t[0] * dims_t[1] * dims_t[2] + idx_t[1] * dims_t[2] + idx_t[2]] =
                    data[idx[0] * dims[1] * dims[2] + idx[1] * dims[2] + idx[2]];
            }
        }
    }
}

struct fill_config {
    static const unsigned n_in = 10;
    static const unsigned fill_value = 0;
};

template <class data_T, typename CONFIG_T> void fill_tensor(data_T data[CONFIG_T::n_in]) {
    #pragma HLS PIPELINE

    for (int i = 0; i < CONFIG_T::n_in; i++) {
        data[i] = (data_T)CONFIG_T::fill_value;
    }
}

} // namespace nnet

#endif
