#ifndef NNET_RECURRENT_H_
#define NNET_RECURRENT_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_recurrent_activation.h"

namespace nnet {

//----------------------
// COMMOM CODE
//----------------------

template<class data_T, class res_T,typename CONFIG_T,class WEIGHT_T>
void multiply_W(data_T input, res_T out[CONFIG_T::n_out], const WEIGHT_T weight[CONFIG_T::n_out]) {

    MULTIPLY_W_LOOP:

    #pragma unroll
    for (int j = 0; j < CONFIG_T::n_out; j++) {
        out[j] = input * weight[j];
        }
    }
template<class data_T, class res_T,typename CONFIG_T,class WEIGHT_T>
void multiply_U(data_T inputs[], res_T out[CONFIG_T::n_out], const WEIGHT_T weight[CONFIG_T::n_out]) {

    MULTIPLY_U_LOOP_I:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_out ; i++) {
        out[i] = 0;
        MULTIPLY_U_LOOP_J:
        #pragma unroll
         for (int j = 0; j < CONFIG_T::n_out; j++) {
            out[i] += (data_T) inputs[j] * weight[j*CONFIG_T::n_out +i];

        }
    }
}

template<class data_T,class res_T, typename CONFIG_T, class WEIGHT_T>
void add_bias(data_T inputs[CONFIG_T::n_out],res_T out[CONFIG_T::n_out], const WEIGHT_T bias[CONFIG_T::n_out]) {

    ADD_BIAS_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_out; i++) {
        out[i] = inputs[i] + bias[i];
    }

}
template<class data_T, class res_T, typename CONFIG_T>
void multiply_vectors(data_T in1[CONFIG_T::n_out], data_T in2[CONFIG_T::n_out], res_T out[CONFIG_T::n_out]) {

    MULTIPLY_VECT_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_out; i++) {
        out[i] = (res_T) in1[i] * in2[i];

    }
}
template<class data_T, class res_T,typename CONFIG_T>
void add_vectors(data_T in1[CONFIG_T::n_out],data_T in2[CONFIG_T::n_out],res_T out[CONFIG_T::n_out]) {

    ADD_VECTOR_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_out; i++) {
        out[i] = (res_T) in1[i] + in2[i];

    }
}

//----------------------
// GRU
//----------------------

struct gru_config {
    // Internal data type definitions
    typedef float weight_t;
    typedef float bias_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in =  1;
    static const unsigned n_out = 1;
    static const unsigned n_units = 1;
    static const unsigned n_timesteps = 1;
    static const unsigned n_outputs = 1;
    static const bool return_sequences = false;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::relu<x_T, y_T, config_T>;
    
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::relu<x_T, y_T, config_T>;
};

template<class data_T, class res_T, typename CONFIG_T>
void gru_cell(
    data_T x[CONFIG_T::n_in],
    res_T  h[CONFIG_T::n_units],
    const typename CONFIG_T::weight_t weights[3 * CONFIG_T::n_units * CONFIG_T::n_in],
    const typename CONFIG_T::weight_t recurrent_weights[3 * CONFIG_T::n_units * CONFIG_T::n_units],
    const typename CONFIG_T::bias_t bias[3 * CONFIG_T::n_units],
    const typename CONFIG_T::bias_t recurrent_bias[3 * CONFIG_T::n_units]
) { 
    static constexpr int recurrent_unroll_factor = CONFIG_T::n_units / CONFIG_T::reuse_factor;
    // A matrix containing the values of matrix product between input (x) and weights (weights), for update, reset and candidate state gates, for each of the units
    hls_register typename CONFIG_T::accum_t mat_mul_x_w[3 * CONFIG_T::n_units];
    nnet::dense_resource<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config_x>(x, mat_mul_x_w, weights, bias);

    // A matrix containing the values of matrix product between previou state (h) and recurrent weights (recurrent_weights), for update, reset and candidate state gates, for each of the units
    hls_register typename CONFIG_T::accum_t mat_mul_h_wr[3 * CONFIG_T::n_units];
    nnet::dense_resource<res_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config_h>(h, mat_mul_h_wr, recurrent_weights, recurrent_bias);

    // A vector containing both the values of z(t) and r(t) for every state 
    hls_register typename CONFIG_T::accum_t z_r [2 * CONFIG_T::n_units]; 
    
    // Add the individual vectors from the multiplication of mat_mul_x_w = Wx*x(t) and mat_mul_h_wr = Wh*h(t-1)
    // Unrolled fully, no DSPs used
    #pragma unroll      
    for(int i = 0; i < (2 * CONFIG_T::n_units); i++) {
        z_r[i] = mat_mul_x_w[i] + mat_mul_h_wr[i];
    }

    // Activation on z(t) and r(t)
    hls_register typename CONFIG_T::accum_t z_r_act [2*CONFIG_T::n_units]; 
    CONFIG_T::template activation_recr<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_RECURRENT_T>::activation(z_r, z_r_act);

    // A matrix containing the values of Hadamard product between r(t) = z_r_act[n_units:2*n_units] and h(t-1) = h
    hls_register typename CONFIG_T::accum_t hadamard_r_h[CONFIG_T::n_units];
    #pragma unroll recurrent_unroll_factor
    for(int i = 0; i < (CONFIG_T::n_units); i++) {
        hadamard_r_h[i] = z_r_act[i + CONFIG_T::n_units] * mat_mul_h_wr[i + 2 * CONFIG_T::n_units];
    }

    // The candidate state; X * W_{hx} + hadmard(r(t), h_(t-1)) * W_{hh} + b_{h}
    typename CONFIG_T::accum_t h_cand[CONFIG_T::n_units];
    // Addition - can unroll fully; no DSPs used here
    #pragma unroll      
    for(int i = 0; i < (CONFIG_T::n_units); i++) {
        h_cand[i] =  mat_mul_x_w[i + 2 * CONFIG_T::n_units] + hadamard_r_h[i];
    }

    // Activation on candidate state
    hls_register typename CONFIG_T::accum_t h_cand_act[CONFIG_T::n_units]; 
    CONFIG_T::template activation<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>::activation(h_cand, h_cand_act);

    // Update state
    #pragma unroll recurrent_unroll_factor
    for(int i = 0; i < (CONFIG_T::n_units); i++) {
        h[i] = static_cast<res_T>(h_cand_act[i] * (1 - z_r_act[i]) + h[i] * z_r_act[i]);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void gru(
    data_T data[CONFIG_T::n_in],
    res_T  res[CONFIG_T::n_outputs * CONFIG_T::n_units],      
    const typename CONFIG_T::weight_t weights[3 * CONFIG_T::n_units * CONFIG_T::n_in],
    const typename CONFIG_T::weight_t recurrent_weights[3 * CONFIG_T::n_units * CONFIG_T::n_units],
    const typename CONFIG_T::bias_t bias[3 * CONFIG_T::n_units],
    const typename CONFIG_T::bias_t recurrent_bias[3 * CONFIG_T::n_units]
) { 

    hls_register data_T x[CONFIG_T::n_in];
    hls_register res_T h[CONFIG_T::n_units];
    
    #pragma unroll
    for(int i = 0; i < CONFIG_T::n_units; i++) {
        h[i] = 0;
    }

    // Loop depedency - cannot pipeline
    #pragma disable_loop_pipelining
    for(int t = 0; t < CONFIG_T::n_timesteps; t++) {
        // Get data at current time step
        #pragma unroll
        for(int j = 0; j < CONFIG_T::n_in; j++) {
            x[j] = data[j + t * CONFIG_T::n_in];
        }
      
        nnet::gru_cell<data_T, res_T, CONFIG_T>(x, h, weights, recurrent_weights, bias, recurrent_bias);

        if (CONFIG_T::return_sequences) {
            #pragma unroll
            for(int i = 0 ; i < CONFIG_T::n_units ; i++) {
                res[CONFIG_T::n_units * t + i] = h[i];
            }
        }
    }
    
    if (!CONFIG_T::return_sequences) {
        #pragma unroll
        for(int i = 0; i < (CONFIG_T::n_units); i++) {
            res[i] = h[i];
        }
    }
}


//----------------------
// SimpleRNN
//----------------------

struct simple_rnn_activ_config {
    static const unsigned n_in = 8;
    static const unsigned table_size = 1024;
    typedef ac_fixed<16,8> table_t;
};

struct simpleRNN_config {
  static const unsigned n_in=1;
  static const unsigned n_out=8;
  static const unsigned n_timestamp=5;
  static const unsigned sliding_window = false;
  static const unsigned return_sequences = false;
  typedef ac_fixed<16,6,true> weight_t;
  typedef ac_fixed<23,3,true> fixed_p_internal_t;
  typedef simple_rnn_activ_config activ_config;

};


template<class data_T, typename CONFIG_T, typename WEIGHT_T>
void simpleRNN_cell(
          data_T hidden_state[CONFIG_T::n_out][CONFIG_T::n_timestamp + 1],
          data_T hidden_state_o[CONFIG_T::n_out],
          data_T inputs,
          const WEIGHT_T kernel[CONFIG_T::n_in*CONFIG_T::n_out],
          const WEIGHT_T rec_kernel[CONFIG_T::n_out*CONFIG_T::n_out],
          const WEIGHT_T bias[CONFIG_T::n_out]){

        //----------------------
        //Internals definitions
        //----------------------

        // Gate outputs

        //Weight multiplication
        typename simpleRNN_config::fixed_p_internal_t afterW[CONFIG_T::n_out] hls_register;
        multiply_W<data_T,simpleRNN_config::fixed_p_internal_t,CONFIG_T,WEIGHT_T>(inputs, afterW, kernel);

        //Bias addition
        typename simpleRNN_config::fixed_p_internal_t afterBias[CONFIG_T::n_out] hls_register;
        add_bias<simpleRNN_config::fixed_p_internal_t,CONFIG_T,WEIGHT_T>(afterW,afterBias, bias);

        //hidden
        typename simpleRNN_config::fixed_p_internal_t hiddenCand[CONFIG_T::n_out] hls_register;
        multiply_U<data_T,simpleRNN_config::fixed_p_internal_t,CONFIG_T,WEIGHT_T>(hidden_state, hiddenCand, rec_kernel);

        typename simpleRNN_config::fixed_p_internal_t afterAdd[CONFIG_T::n_out];
        add_vectors<simpleRNN_config::fixed_p_internal_t, CONFIG_T>(afterBias, hiddenCand, afterAdd);

        data_T h[CONFIG_T::n_out];

        //Activation
        //hls_fpga insert activation

       OUTPUT_WRITE_LOOP:
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
          hidden_state_o[x]=h[x];
        }
        return;
}

template<class data_T, class res_T, typename CONFIG_T, class WEIGHT_T>
  void simple_rnn_network(data_T input0[CONFIG_T::n_timestamp*CONFIG_T::n_in], res_T res[CONFIG_T::n_timestamp*CONFIG_T::n_out],
  const WEIGHT_T kernel[CONFIG_T::n_in*CONFIG_T::n_out], const WEIGHT_T rec_kernel[CONFIG_T::n_out*CONFIG_T::n_out], const WEIGHT_T bias[CONFIG_T::n_out]){

    data_T hidden_state[CONFIG_T::n_out][CONFIG_T::n_timestamp + 1] hls_register;
    data_T hidden_state_temp[CONFIG_T::n_out] hls_register;
    data_T h[CONFIG_T::n_out] hls_register;

    static data_T inputs[CONFIG_T::n_timestamp*CONFIG_T::n_in] hls_register;

    INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_out; x++) {
      hidden_state[x][0]=0;
    }

    #pragma unroll
    #pragma ivdep
 
    //Input dimention

      for (int j=0; j<CONFIG_T::n_timestamp; j++){
        for (int z=0; z<CONFIG_T::n_in; z++){
          inputs[z* CONFIG_T::n_in + j] = input0[z * CONFIG_T::n_in + j];
        }
      }

    #pragma unroll 
    for (int i=0; i < CONFIG_T::n_timestamp; i++){
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state_temp[x] = hidden_state[x][i];
      }

      for (int j=0; j<CONFIG_T::n_in; j++){
        simpleRNN_cell<data_T,CONFIG_T,WEIGHT_T>(hidden_state_temp,h,inputs[i], kernel, rec_kernel, bias);
      }

      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state[x][i+1]=h[x];
      }
    }


    if(CONFIG_T::return_sequences == 0){
      //Output when return_sequences is false 
      #pragma unroll           
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        res[x]= hidden_state[x][CONFIG_T::n_timestamp];
      }
    }
    else{
      //Output when return_sequences is true
      #pragma unroll
      for(int x = 0; x < CONFIG_T::n_timestamp; x++){ 
        for(int h = 0; h < CONFIG_T::n_out; h++){
            res[x + h * CONFIG_T::n_out ] = hidden_state[h][x+1];
        }
      }
    }
  }

template<class data_T, class res_T, typename CONFIG_T, class WEIGHT_T>
  void simple_rnn_network(data_T input0, res_T res[CONFIG_T::n_timestamp*CONFIG_T::n_out],
  const WEIGHT_T kernel[CONFIG_T::n_in*CONFIG_T::n_out], const WEIGHT_T rec_kernel[CONFIG_T::n_out*CONFIG_T::n_out], const WEIGHT_T bias[CONFIG_T::n_out]){

    data_T hidden_state[CONFIG_T::n_out][CONFIG_T::n_timestamp + 1] hls_register;
    data_T hidden_state_temp[CONFIG_T::n_out] hls_register;
    data_T h[CONFIG_T::n_out] hls_register;

    static data_T inputs[CONFIG_T::n_timestamp] hls_register;

    INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_out; x++) {
      hidden_state[x][0]=0;
    }

    #pragma unroll
    #pragma ivdep

    for (int j=1;j<CONFIG_T::n_timestamp; j++){
      inputs[j-1] = inputs[j];
    }
    inputs[CONFIG_T::n_timestamp-1]=input0;

    #pragma unroll 
    for (int i=0; i < CONFIG_T::n_timestamp; i++){
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state_temp[x] = hidden_state[x][i];
      }

      simpleRNN_cell<data_T,CONFIG_T,WEIGHT_T>(hidden_state_temp,h,inputs[i], kernel, rec_kernel, bias);
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state[x][i+1]=h[x];
      }
    }


    if(CONFIG_T::return_sequences == 0){
      //Output when return_sequences is false
      #pragma unroll            
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        res[x]= hidden_state[x][CONFIG_T::n_timestamp];
      }
    }
    else{
      //Output when return_sequences is true
      #pragma unroll
      for(int x = 0; x < CONFIG_T::n_timestamp; x++){ 
        for(int h = 0; h < CONFIG_T::n_out; h++){
            res[x + h * CONFIG_T::n_out ] = hidden_state[h][x+1];
        }
      }
    }
  }

//----------------------
// LSTM 
//----------------------

struct lstm_activ_config {
    static const unsigned n_in = 10;
    static const unsigned table_size = 1024;
    typedef ac_fixed<16,8> table_t;
};

struct lstm_config {
  static const unsigned n_in=1;
  static const unsigned n_out=10;
  static const unsigned sliding_window = false;
  static const unsigned return_sequences = false;
  typedef ac_fixed<16,6,true> weight_t;
  typedef lstm_activ_config activ_config;

};

template<class data_T, typename CONFIG_T, typename WEIGHT_T>
  void lstm_cell(
        data_T hidden_state[CONFIG_T::n_out][CONFIG_T::n_timestamp + 1],
        data_T hidden_state_o[CONFIG_T::n_out],
        data_T cell_state[CONFIG_T::n_out][CONFIG_T::n_timestamp + 1],
        data_T cell_state_o[CONFIG_T::n_out],
        data_T inputs,
        const WEIGHT_T WI[CONFIG_T::n_in*CONFIG_T::n_out], const WEIGHT_T WF[CONFIG_T::n_in*CONFIG_T::n_out], const WEIGHT_T WC[CONFIG_T::n_in*CONFIG_T::n_out], const WEIGHT_T WO[CONFIG_T::n_in*CONFIG_T::n_out],
        const WEIGHT_T RWI[CONFIG_T::n_out*CONFIG_T::n_out], const WEIGHT_T RWF[CONFIG_T::n_out*CONFIG_T::n_out], const WEIGHT_T RWC[CONFIG_T::n_out*CONFIG_T::n_out], const WEIGHT_T RWO[CONFIG_T::n_out*CONFIG_T::n_out],
        const WEIGHT_T BI[CONFIG_T::n_out], const WEIGHT_T BF[CONFIG_T::n_out], const WEIGHT_T BC[CONFIG_T::n_out], const WEIGHT_T BO[CONFIG_T::n_out]){

    //----------------------
    //Internals definitions
    //----------------------

    data_T i_afterW   [CONFIG_T::n_out] ;
    data_T i_afterBias[CONFIG_T::n_out] ;
    data_T c_afterW   [CONFIG_T::n_out] ;
    data_T c_afterBias[CONFIG_T::n_out] ;
    data_T o_afterW   [CONFIG_T::n_out] ;
    data_T o_afterBias[CONFIG_T::n_out] ;
    data_T f_afterW   [CONFIG_T::n_out] ;
    data_T f_afterBias[CONFIG_T::n_out] ;

    // Hidden state Gate candidates, intermediate variables
    data_T i_hiddenCand[CONFIG_T::n_out] ;
    data_T f_hiddenCand[CONFIG_T::n_out] ;
    data_T c_hiddenCand[CONFIG_T::n_out] ;
    data_T o_hiddenCand[CONFIG_T::n_out] ;

    // AfterAddition, intermediate variables
    data_T i_afterAdd[CONFIG_T::n_out] ;
    data_T f_afterAdd[CONFIG_T::n_out] ;
    data_T c_afterAdd[CONFIG_T::n_out] ;
    data_T o_afterAdd[CONFIG_T::n_out] ;

    // Gate outputs
    data_T gate_i[CONFIG_T::n_out] ;
    data_T gate_f[CONFIG_T::n_out] ;
    data_T gate_c[CONFIG_T::n_out] ;
    data_T gate_o[CONFIG_T::n_out] ;
    data_T gate_ic[CONFIG_T::n_out] ;
    data_T gate_forget[CONFIG_T::n_out] ;

    data_T h[CONFIG_T::n_out] ;


    //intermediate variable cell calculation
    data_T cell_act_multp[CONFIG_T::n_out] ;
    data_T cell_act_add[CONFIG_T::n_out] ;


    //-----------Gate I Calculations
    //Weight multiplication
    multiply_W<data_T,data_T,CONFIG_T,WEIGHT_T>(inputs, i_afterW , WI);
    //Bias addition
    add_bias<data_T,data_T,CONFIG_T,WEIGHT_T>(i_afterW, i_afterBias, BI);
    //Hidden Candidate
    multiply_U<data_T,data_T,CONFIG_T,WEIGHT_T>(hidden_state, i_hiddenCand, RWI);
    add_vectors<data_T,data_T,CONFIG_T>(i_afterBias, i_hiddenCand, i_afterAdd);
    //Activation
    //hls_fpga insert recurrent_activation --- Gate I


    //-----------Gate F Calculations
    //Weight multiplication
    multiply_W<data_T,data_T,CONFIG_T,WEIGHT_T>(inputs, f_afterW, WF);
    //Bias addition
    add_bias<data_T,data_T,CONFIG_T,WEIGHT_T>(f_afterW, f_afterBias, BF);
    //Hidden Candidate
    multiply_U<data_T,data_T,CONFIG_T,WEIGHT_T>(hidden_state, f_hiddenCand, RWF);
    add_vectors<data_T,data_T,CONFIG_T>(f_afterBias, f_hiddenCand, f_afterAdd);
    //Activation
    //hls_fpga insert recurrent_activation --- Gate F


    //-----------Gate C Calculations
    //Weight multiplication
    multiply_W<data_T,data_T,CONFIG_T,WEIGHT_T>(inputs, c_afterW, WC);
    //Bias addition
    add_bias<data_T,data_T,CONFIG_T,WEIGHT_T>(c_afterW, c_afterBias, BC);
    //Hidden Candidate
    multiply_U<data_T,data_T,CONFIG_T,WEIGHT_T>(hidden_state, c_hiddenCand, RWC);
    add_vectors<data_T,data_T,CONFIG_T>(c_afterBias, c_hiddenCand, c_afterAdd);
    //Activation
    //hls_fpga insert activation  --- Gate C


    //-----------gate I and C multiply
    multiply_vectors<data_T,data_T,CONFIG_T>(gate_i, gate_c, gate_ic);

    //-----------Gate O Calculations
    multiply_W<data_T,data_T,CONFIG_T,WEIGHT_T>(inputs, o_afterW, WO);
    add_bias<data_T,data_T,CONFIG_T,WEIGHT_T>(o_afterW, o_afterBias, BO);
    multiply_U<data_T,data_T,CONFIG_T,WEIGHT_T>(hidden_state, o_hiddenCand, RWO);
    add_vectors<data_T,data_T,CONFIG_T>(o_afterBias, o_hiddenCand, o_afterAdd);
    //hls_fpga insert recurrent_activation  --- Gate O


    //-----------Cell State Calculation
    multiply_vectors<data_T,data_T,CONFIG_T>(gate_f, cell_state, cell_act_multp);
    add_vectors<data_T,data_T,CONFIG_T>(gate_ic, cell_act_multp, cell_act_add);

    //-----------Forget gate Calculation
    //hls_fpga insert activation  --- Forget Gate

    multiply_vectors<data_T,data_T,CONFIG_T>(gate_o, gate_forget, h);


    OUTPUT_WRITE_LOOP:
    #pragma unroll
    for (int x = (CONFIG_T::n_out - 1); x >= 0; x--) {
      hidden_state_o[x]=h[x];
      cell_state_o[x]=cell_act_add[x];
    }
    return;
  }


template<class data_T, class res_T,class CONFIG_T ,class WEIGHT_T>
  void lstm_network( data_T input0[CONFIG_T::n_timestamp*CONFIG_T::n_in], res_T res[CONFIG_T::n_timestamp*CONFIG_T::n_out],
            const WEIGHT_T WI[CONFIG_T::n_in*CONFIG_T::n_out], const WEIGHT_T WF[CONFIG_T::n_in*CONFIG_T::n_out], const WEIGHT_T WC[CONFIG_T::n_in*CONFIG_T::n_out], const WEIGHT_T WO[CONFIG_T::n_in*CONFIG_T::n_out],
            const WEIGHT_T RWI[CONFIG_T::n_out*CONFIG_T::n_out], const WEIGHT_T RWF[CONFIG_T::n_out*CONFIG_T::n_out], const WEIGHT_T RWC[CONFIG_T::n_out*CONFIG_T::n_out], const WEIGHT_T RWO[CONFIG_T::n_out*CONFIG_T::n_out],
            const WEIGHT_T BI[CONFIG_T::n_out], const WEIGHT_T BF[CONFIG_T::n_out], const WEIGHT_T BC[CONFIG_T::n_out], const WEIGHT_T BO[CONFIG_T::n_out]){

    data_T hidden_state[CONFIG_T::n_out][CONFIG_T::n_timestamp + 1] hls_register;
    data_T cell_state  [CONFIG_T::n_out][CONFIG_T::n_timestamp + 1] hls_register;
    data_T hidden_state_temp[CONFIG_T::n_out] hls_register;
    data_T cell_state_temp  [CONFIG_T::n_out] hls_register;
    data_T h[CONFIG_T::n_out] hls_register;
    data_T c[CONFIG_T::n_out] hls_register;

    static data_T inputs[CONFIG_T::n_timestamp*CONFIG_T::n_in] hls_register;
    
    INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_out; x++) {
      hidden_state[x][0]=0;
      cell_state[x][0]=0;
    }

    #pragma unroll
    #pragma ivdep

    //Input dimention

      for (int j=0; j<CONFIG_T::n_timestamp; j++){
        for (int z=0; z<CONFIG_T::n_in; z++){
          inputs[z* CONFIG_T::n_in + j] = input0[z * CONFIG_T::n_in + j];
        }
      }

    #pragma unroll 
    for (int i=0; i < CONFIG_T::n_timestamp; i++){
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state_temp[x] = hidden_state[x][i];
        cell_state_temp[x]   = cell_state[x][i];
      }

      for (int j=0; j<CONFIG_T::n_in; j++){
        lstm_cell<data_T,CONFIG_T,WEIGHT_T>(hidden_state_temp,h,cell_state_temp,c,inputs[i + j* CONFIG_T::n_in],WI,WF,WC,WO,RWI,RWF,RWC,RWO,BI,BF,BC,BO);
      }
    
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state[x][i+1]=h[x];
        cell_state[x][i+1]=c[x];
      }
    }


    if(CONFIG_T::return_sequences == 0){
      //Output when return_sequences is false    
      #pragma unroll        
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        res[x]= hidden_state[x][CONFIG_T::n_timestamp];
      }
    }
    else{
      //Output when return_sequences is true
      #pragma unroll
      for(int x = 0; x < CONFIG_T::n_timestamp; x++){ 
        for(int h = 0; h < CONFIG_T::n_out; h++){
            res[x + h * CONFIG_T::n_out ] = hidden_state[h][x+1];
        }
      }
    }
  }

  template<class data_T, class res_T,class CONFIG_T ,class WEIGHT_T>
  void lstm_network(data_T input0,res_T res[CONFIG_T::n_timestamp*CONFIG_T::n_out],
            const WEIGHT_T WI[CONFIG_T::n_in*CONFIG_T::n_out], const WEIGHT_T WF[CONFIG_T::n_in*CONFIG_T::n_out], const WEIGHT_T WC[CONFIG_T::n_in*CONFIG_T::n_out], const WEIGHT_T WO[CONFIG_T::n_in*CONFIG_T::n_out],
            const WEIGHT_T RWI[CONFIG_T::n_out*CONFIG_T::n_out], const WEIGHT_T RWF[CONFIG_T::n_out*CONFIG_T::n_out], const WEIGHT_T RWC[CONFIG_T::n_out*CONFIG_T::n_out], const WEIGHT_T RWO[CONFIG_T::n_out*CONFIG_T::n_out],
            const WEIGHT_T BI[CONFIG_T::n_out], const WEIGHT_T BF[CONFIG_T::n_out], const WEIGHT_T BC[CONFIG_T::n_out], const WEIGHT_T BO[CONFIG_T::n_out]){

    data_T hidden_state[CONFIG_T::n_out][CONFIG_T::n_timestamp + 1] hls_register;
    data_T cell_state  [CONFIG_T::n_out][CONFIG_T::n_timestamp + 1] hls_register;
    data_T hidden_state_temp[CONFIG_T::n_out] hls_register;
    data_T cell_state_temp  [CONFIG_T::n_out] hls_register;
    data_T h[CONFIG_T::n_out] hls_register;
    data_T c[CONFIG_T::n_out] hls_register;

    static data_T inputs[CONFIG_T::n_timestamp] hls_register;

    INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_out; x++) {
      hidden_state[x][0]=0;
      cell_state[x][0]=0;
    }

    #pragma unroll
    #pragma ivdep
    for (int j=1;j<CONFIG_T::n_timestamp; j++){
      inputs[j-1] = inputs[j];
    }
    inputs[CONFIG_T::n_timestamp-1]=input0;

    #pragma unroll 
    for (int i=0; i < CONFIG_T::n_timestamp; i++){
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state_temp[x] = hidden_state[x][i];
        cell_state_temp[x]   = cell_state[x][i];
      }

      lstm_cell<data_T,CONFIG_T,WEIGHT_T>(hidden_state_temp,h,cell_state_temp,c,inputs[i],WI,WF,WC,WO,RWI,RWF,RWC,RWO,BI,BF,BC,BO);

      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state[x][i+1]=h[x];
        cell_state[x][i+1]=c[x];
      }
    }


    if(CONFIG_T::return_sequences == 0){
      //Output when return_sequences is false  
      #pragma unroll          
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        res[x]= hidden_state[x][CONFIG_T::n_timestamp];
      }
    }
    else{
      //Output when return_sequences is true
      #pragma unroll
      for(int x = 0; x < CONFIG_T::n_timestamp; x++){ 
        for(int h = 0; h < CONFIG_T::n_out; h++){
            res[x + h * CONFIG_T::n_out ] = hidden_state[h][x+1];
        }
      }
    }
  }




}

#endif
