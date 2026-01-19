#include "qsoftmax.h"

#include <stdio.h>

int main(void) {
  float x[] = {1.0f, 2.0f, -1.0f, 0.5f};
  const int n = (int)(sizeof(x) / sizeof(x[0]));

  float y[n];
  qsoftmax_params p = qsoftmax_default_params();
  qsoftmax_poly5_div(x, n, y, p);

  float s = 0.0f;
  for (int i = 0; i < n; i++) {
    printf("y[%d]=%.9f\n", i, y[i]);
    s += y[i];
  }
  printf("sum=%.9f\n", s);
  return 0;
}

