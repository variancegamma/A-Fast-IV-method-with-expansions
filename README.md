# A-Fast-IV-method-with-analytical-expansions

This repo provides the underlying code for a fast implied-volatility method based on analytical expansions.

The following compiler command is used across all benchmarks in the paper:

```bash
gcc -Ofast -march=native -mtune=native -ffp-contract=fast \
    -fno-math-errno -fno-trapping-math -funsafe-math-optimizations \
    -fno-signed-zeros -fno-rounding-math -fomit-frame-pointer \
    -o bench_iv_c_all_hh4.exe bench_iv_c_all_hh4.c -lm
```
After GCC compilation, the executable bench_iv_c_all_hh4.exe is ready to be run standalone.


## Benchmark Results

Timings report the best of 5 runs. The standard grid contains 328 base points repeated 5,000 times, giving 1,640,000 implied-volatility inversions.

### Grid definitions

| Grid | Volatility points | Delta points | Base points | Total IV inversions |
|---|---|---|---:|---:|
| `328-p` | `v ∈ {0.01, 0.05, 0.10, ..., 2.00}` | `Δ ∈ {0.05, 0.20, 0.30, 0.45, 0.55, 0.70, 0.80, 0.95}` | 328 | 1,640,000 |
| `1970-p` | `v ∈ {0.01, 0.05, 0.06, ..., 2.00}` | `Δ ∈ {0.01, 0.05, 0.20, 0.30, 0.45, 0.55, 0.70, 0.80, 0.95, 0.99}` | 1,970 | 9,850,000 |

### Cross-platform benchmark

| Compiler | Grid | Method | ns/IV | Max error | Speedup |
|---|---:|---|---:|---:|---:|
| MSYS2 UCRT64 GCC 16.1 | 328-p | Jäckel reference implementation | 197.9 | 1.679e-15 | — |
| MSYS2 UCRT64 GCC 16.1 | 328-p | Regime-split HH-4 | 112.7-114.2 | 8.52e-14 | **1.733-1.756x** |
| MSYS2 UCRT64 GCC 16.1 | 1970-p | Jäckel reference implementation | 208.4 | 7.327e-15 | — |
| MSYS2 UCRT64 GCC 16.1 | 1970-p | Regime-split HH-4 | 124.9 | 6.026e-13 | **1.668x** |

### Standalone C benchmark

This benchmark removes the Python/ctypes shared-library wrapper used in the Jäckel comparison and times the Regime-split HH-4 implementation directly as a standalone C executable.

| Implementation | Grid | Best ns/IV | Median ns/IV | Maximum error |
|---|---:|---:|---:|---:|
| Regime-split HH-4, standalone C | 1,640,000 | 117.2 | 118.8 | 7.333e-14 |

The Jackel benchmark can be replicated using simple python script run.
```bash
python bench_jackel_vs_regime.py
```
