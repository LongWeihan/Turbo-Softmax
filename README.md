# Turbo-Softmax

A high-precision, yet blazing fast Softmax implementation in C.

It achieves speed without sacrificing stability by using **Range Reduction** ($2^i$, $e^t$):

* **$2^i$**: Constructed directly via **IEEE-754 bit-level manipulation**.
* **$e^t$**: Approximated using a **5th-order polynomial**.
* **Normalization**: Uses direct division (`1.0f / sum`) for accuracy.

## ⚡ Performance Highlights

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Speedup** | **4.0x - 4.2x** | Consistent across dims 16-1024 |
| **Throughput** | **~330 MElem/s** | vs. ~80 MElem/s (Reference) |
| **Top-1 Accuracy** | **100% Match** | 0.0% Mismatch rate |
| **Top-5 Accuracy** | **100% Match** | 0.0% Mismatch rate |
| **Top-10 Accuracy** | **100% Match** | 0.0% Mismatch rate |
| **Max Error** | **< 1e-6** | Negligible KL-Divergence (< 5e-9) |



<img width="989" height="841" alt="image" src="https://github.com/user-attachments/assets/9739ddd1-fb3c-4434-9251-e07af4c14ecd" />
<img width="996" height="349" alt="image" src="https://github.com/user-attachments/assets/b61447fb-8a79-436e-a53f-3eb7c11f3b54" />



## Comparison with Other Softmax Acceleration Schemes

### Summary Comparison Table

| Scheme | Speed | Precision | Memory Overhead | Hardware Dependency | Best Use Case |
| --- | --- | --- | --- | --- | --- |
| Standard Lib (`math.h`) | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | None | Servers, PC, Precision-critical tasks |
| Look-Up Table (LUT) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | None | Legacy MCUs, devices without FPU |
| SIMD / DSP Libs (NEON/AVX) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | **High** (Arch specific) | Mobile phones, Raspberry Pi, High-perf Embedded |
| Quantization (Int8) | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Medium | NPUs, Ultra-low power, Battery devices |
| **Turbo-Softmax** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Low** (Generic C) | **General Purpose MCUs (STM32, ESP32, RISC-V)** |

### Verdict

**Turbo-Softmax** is the ideal choice when:

1. You are running **Float32 inference** on a microcontroller.
2. You cannot afford the memory footprint of a large Look-Up Table.
3. Your hardware lacks advanced vector (SIMD) instructions, or you need a **cross-platform** solution that "just works."

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
