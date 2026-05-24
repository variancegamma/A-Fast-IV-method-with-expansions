# A-Fast-IV-method-with-analytical-expansions

This repo provides the underlying code for a fast implied-volatility method based on analytical expansions.

The following compiler command is used across all benchmarks in the paper:

```bash
gcc -Ofast -march=native -mtune=native -ffp-contract=fast \
    -fno-math-errno -fno-trapping-math -funsafe-math-optimizations \
    -fno-signed-zeros -fno-rounding-math -fomit-frame-pointer \
    -o bench_iv_c_all_hh4.exe bench_iv_c_all_hh4.c -lm
