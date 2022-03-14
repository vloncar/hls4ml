#ifndef OP_UTILS_H_
#define OP_UTILS_H_

#include "hls_stream.h"

template<class data_T, size_t N>
void copy_input_array(
    const float data[N],
    data_T ap_data[N],
    unsigned batch
) {
    CopyInput: for(int i = 0; i < N; i++) {
        ap_data[i] = (data_T) data[batch * N + i];
    }
}

template<class data_T, size_t N>
void copy_input_stream(
    const float data[N],
    hls::stream<data_T> &ap_data,
    unsigned batch
) {
    CopyInput: for(int i_in = 0; i_in < N / data_T::size; i_in++) {
        data_T data_pack;
        DataPack: for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
            data_pack[i_pack] = data[batch * N + i_in * data_T::size + i_pack];
        }
        ap_data.write(data_pack);
    }
}

template<class res_T, size_t N>
void copy_result_array(
    const res_T ap_result[N],
    float result[N],    
    unsigned batch
) {
    CopyResult: for(int r = 0; r < N; r++) {
        result[batch * N + r] = ap_result[r].to_float();
    }
}

template<class res_T, size_t N>
void copy_result_stream(
    hls::stream<res_T> &ap_result,
    float result[N],    
    unsigned batch
) {
    CopyResult: for(int i_out = 0; i_out < N / res_T::size; i_out++) {
        res_T res_pack = ap_result.read();
        DataPack: for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            result[batch * N + i_out * res_T::size + i_pack] = res_pack[i_pack];
        }
    }
}

#endif // OP_UTILS_H_