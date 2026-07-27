#ifndef NNET_INSTR_GEN_H_
#define NNET_INSTR_GEN_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_function_stubs.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"

namespace nnet {

// Per-layer generated code (e.g. conv line buffers) is inserted below by the writer's
// write_generated_code pass. Dense-only models generate nothing here.

// hls4ml insert code

} // namespace nnet

#endif
