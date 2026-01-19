#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qsoftmax_params {
  // Clamp (x - max) to [clamp_min_x, 0]. Default -80 is a safe choice for float32.
  float clamp_min_x;
} qsoftmax_params;

qsoftmax_params qsoftmax_default_params(void);

// High-accuracy fast softmax: exp via poly5 + IEEE-754 2^i, normalize via division.
void qsoftmax_poly5_div(const float* x, int n, float* out, qsoftmax_params p);

// Reference implementation (stable softmax with expf), useful for verification.
void softmax_ref(const float* x, int n, float* out);

#ifdef __cplusplus
}  // extern "C"
#endif

