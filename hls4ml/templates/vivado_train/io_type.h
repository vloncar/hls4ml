#include "op_utils.h"

#ifndef IO_TYPE
    #define IO_TYPE io_parallel
#endif

#if IO_TYPE == io_parallel
    template <class data_T, size_t N_IN>
    using ap_data_t = data_T[N_IN]; // Oof!

    template <class res_T, size_t N_OUT>
    using ap_res_t = res_T[N_OUT];

    template<class data_T, size_t N_IN>
    const auto copy_input = copy_input_array<data_T, N_IN>;

    template<class res_T, size_t N_OUT>
    const auto copy_result = copy_result_array<res_T, N_OUT>;

#elif IO_TYPE == io_stream
    template <class data_T, size_t N_IN>
    using ap_data_t = hls::stream<data_T>;

    template <class res_T, size_t N_OUT>
    using ap_res_t = hls::stream<res_T>;

    template<class data_T, size_t N_IN>
    const auto copy_input = copy_input_stream<data_T, N_IN>;

    template<class res_T, size_t N_OUT>
    const auto copy_result = copy_result_stream<res_T, N_OUT>;

#else
    #error "Unexpected value of IO_TYPE."
#endif