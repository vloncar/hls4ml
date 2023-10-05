#ifndef NNET_SCATTER_H_
#define NNET_SCATTER_H_

#include "nnet_helpers.h"

namespace nnet {

// Return the sum of the two inputs
template <typename T> T scatter_sum(T x, T y) { return x + y; }

// Return the maximum of the two inputs
template <typename T> T scatter_max(T x, T y) { return (x > y) ? x : y; }

// Return the minimum of the two inputs
template <typename T> T scatter_min(T x, T y) { return (x < y) ? x : y; }

// Return the product of the two inputs
template <typename T> T scatter_mul(T x, T y) { return x * y; }

// Enumeration for scatter operation (sum, max, min, mul)
enum Scatter_Op { ScatterSum, ScatterMax, ScatterMin, ScatterMul }; // TODO Add mean
template <typename T, Scatter_Op op> T scatter_op(T x, T y) {
    switch (op) {
    case ScatterSum:
        return scatter_sum<T>(x, y);
    case ScatterMax:
        return scatter_max<T>(x, y);
    case ScatterMin:
        return scatter_min<T>(x, y);
    case ScatterMul:
        return scatter_mul<T>(x, y);
    }
}

template <typename T, Scatter_Op op> T init_val() {
    switch (op) {
    case ScatterSum:
        return 0;
    case ScatterMax: {
        T x = 0;
        x[x.width - 1] = 1;
        return x;
    }
    case ScatterMin: {
        T x = 0;
        x[x.width - 1] = 1;
        return ~x;
    }
    case ScatterMul:
        return 1;
    }
}

struct scatter_config_1d {
    static const unsigned n_in = 5;
    static const unsigned n_index = 5;
    static const unsigned n_out = 5;
    static const bool init_output = false;
    static const Scatter_Op scatter_op = ScatterSum;
};

template <class data_T, class index_T, class target_T, class res_T, typename CONFIG_T>
void scatter_1d(data_T src[CONFIG_T::n_in], index_T index[CONFIG_T::n_index], target_T target[CONFIG_T::n_out],
                res_T result[CONFIG_T::n_out]) {

    // Prepare tensor array with initial value or values from target tensor
    if (CONFIG_T::init_output) {
        for (int i = 0; i < CONFIG_T::n_out; i++) {
            result[i] = init_val<res_T, CONFIG_T::scatter_op>();
        }
    } else {
        for (int i = 0; i < CONFIG_T::n_out; i++) {
            result[i] = target[i];
        }
    }

    // Perform scatter operation
    for (int i = 0; i < CONFIG_T::n_index; i++) {
        res_T res = result[index[i]];
        result[index[i]] = scatter_op<data_T, CONFIG_T::scatter_op>(res, src[i]);
    }
}

struct scatter_config_2d {
    static const unsigned n_in_0 = 5;
    static const unsigned n_in_1 = 5;
    static const unsigned n_index_0 = 5;
    static const unsigned n_index_1 = 5;
    static const unsigned n_out_0 = 5;
    static const unsigned n_out_1 = 5;
    static const unsigned dim = 1;
    static const bool init_output = false;
    static const Scatter_Op scatter_op = ScatterSum;
};

template <class data_T, class index_T, class target_T, class res_T, typename CONFIG_T>
void scatter_2d(data_T src[CONFIG_T::n_in_0 * CONFIG_T::n_in_1], index_T index[CONFIG_T::n_index_0 * CONFIG_T::n_index_1],
                target_T target[CONFIG_T::n_out_0 * CONFIG_T::n_out_1],
                res_T result[CONFIG_T::n_out_0 * CONFIG_T::n_out_1]) {

    // Prepare tensor array with initial value or values from target tensor
    if (CONFIG_T::init_output) {
        for (int i = 0; i < CONFIG_T::n_out_0; i++) {
            for (int j = 0; j < CONFIG_T::n_out_1; j++) {
                result[i * CONFIG_T::n_out_1 + j] = init_val<res_T, CONFIG_T::scatter_op>();
            }
        }
    } else {
        for (int i = 0; i < CONFIG_T::n_out_0; i++) {
            for (int j = 0; j < CONFIG_T::n_out_1; j++) {
                result[i * CONFIG_T::n_out_1 + j] = target[i * CONFIG_T::n_out_1 + j];
            }
        }
    }

    // Perform scatter operation
    for (int i = 0; i < CONFIG_T::n_index_0; ++i) {
        for (int j = 0; j < CONFIG_T::n_index_1; ++j) {
            if (CONFIG_T::dim == 0) {
                int idx = CONFIG_T::n_out_1 * index[i * CONFIG_T::n_index_1 + j] + j;
                result[idx] = scatter_op<data_T, CONFIG_T::scatter_op>(result[idx], src[i * CONFIG_T::n_in_1 + j]);
            } else if (CONFIG_T::dim == 1 || CONFIG_T::dim == -1) {
                int idx = i * CONFIG_T::n_out_1 + index[i * CONFIG_T::n_index_1 + j];
                result[idx] = scatter_op<data_T, CONFIG_T::scatter_op>(result[idx], src[i * CONFIG_T::n_in_1 + j]);
            }
        }
    }
}

} // namespace nnet

#endif
