#include "qsoftmax.h"

#include <math.h>
#include <string.h>

static inline float u32_to_f32(uint32_t u) {
  float x;
  memcpy(&x, &u, sizeof(x));
  return x;
}

qsoftmax_params qsoftmax_default_params(void) {
  qsoftmax_params p;
  p.clamp_min_x = -80.0f;
  return p;
}

static inline float clamp_x(float x, float clamp_min_x) {
  if (x < clamp_min_x) {
    x = clamp_min_x;
  }
  if (x > 0.0f) {
    x = 0.0f;
  }
  return x;
}

static inline float fast_exp_poly5(float x, float clamp_min_x) {
  x = clamp_x(x, clamp_min_x);

  // Range reduction: exp(x) = 2^y, y = x * log2(e) = i + f, f in [-0.5, 0.5].
  const float LOG2_E = 1.44269504088896341f;
  const float LN2 = 0.69314718055994531f;

  float y = x * LOG2_E;
  int i = (int)(y + (y >= 0.0f ? 0.5f : -0.5f));
  float f = y - (float)i;

  if (i < -126) {
    return 0.0f;
  }
  if (i > 127) {
    i = 127;
  }

  float t = f * LN2;

  // exp(t) ≈ 1 + t + t^2/2 + t^3/6 + t^4/24 + t^5/120
  float p = 1.0f + t * (1.0f + t * (0.5f + t * (0.1666666716f + t * (0.0416666679f + t * 0.00833333377f))));

  float two_i = u32_to_f32((uint32_t)(i + 127) << 23);
  return two_i * p;
}

void qsoftmax_poly5_div(const float* x, int n, float* out, qsoftmax_params p) {
  if (n <= 0) {
    return;
  }

  float max_x = x[0];
  for (int i = 1; i < n; i++) {
    if (x[i] > max_x) {
      max_x = x[i];
    }
  }

  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    float e = fast_exp_poly5(x[i] - max_x, p.clamp_min_x);
    out[i] = e;
    sum += e;
  }

  float inv_sum = 1.0f / sum;
  for (int i = 0; i < n; i++) {
    out[i] *= inv_sum;
  }
}

void softmax_ref(const float* x, int n, float* out) {
  if (n <= 0) {
    return;
  }

  float max_x = x[0];
  for (int i = 1; i < n; i++) {
    if (x[i] > max_x) {
      max_x = x[i];
    }
  }

  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    float e = expf(x[i] - max_x);
    out[i] = e;
    sum += e;
  }

  float inv_sum = 1.0f / sum;
  for (int i = 0; i < n; i++) {
    out[i] *= inv_sum;
  }
}
