# Turbo-Softmax

A "high-precision, yet blazing fast" Softmax implementation in C.

It achieves speed without sacrificing stability by using **Range Reduction** ($2^i$, $e^t$):

* **$2^i$**: Constructed directly via **IEEE-754 bit-level manipulation**.
* **$e^t$**: Approximated using a **5th-order polynomial**.
* **Normalization**: Uses direct division (`1.0f / sum`) for accuracy.

## 📂 Files

* `qsoftmax.h` / `qsoftmax.c`: Core implementation (contains `poly5` approximation + `div` normalization).
* `example.c`: Minimal usage example.

## 🚀 Build & Run

To compile with optimizations (recommended for benchmarking) using GCC/MinGW:

```powershell
gcc -O3 -std=c11 -Wall -Wextra -pedantic example.c qsoftmax.c -lm -o example.exe
./example.exe

```

## 📚 API Reference

```c
typedef struct qsoftmax_params {
  float clamp_min_x; // Threshold for minimal x (Default: -80.0f)
} qsoftmax_params;

// Initialize default parameters
qsoftmax_params qsoftmax_default_params(void);

/**
 * Compute softmax using 5th-order polynomial approximation.
 * * @param x   Input array
 * @param n   Number of elements
 * @param out Output array (can be the same as x)
 * @param p   Parameters (use qsoftmax_default_params())
 */
void qsoftmax_poly5_div(const float* x, int n, float* out, qsoftmax_params p);

```
