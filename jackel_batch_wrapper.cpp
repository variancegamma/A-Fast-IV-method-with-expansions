// jackel_batch_wrapper.cpp
//
// Batch wrapper around Peter Jäckel's LetsBeRational normalised IV routine.
//
// IMPORTANT: LetsBeRational's importexport.h already defines EXPORT on Windows
// as __declspec(dllexport). We use WRAP_EXPORT below to avoid the name clash.
//
// Compile with the original LBR sources — run bench_jackel_vs_regime.py
// which handles this automatically, OR manually:
//
//   Linux/macOS:
//     g++ -O3 -march=native -ffast-math -DNDEBUG -shared -fPIC \
//         -I<lbr_src> -o iv_jackel_original.so \
//         jackel_batch_wrapper.cpp LetsBeRational.cpp erf_cody.cpp \
//         rationalcubic.cpp normaldistribution.cpp -lm
//
//   Windows (MinGW):
//     g++ -O3 -march=native -ffast-math -DNDEBUG -shared \
//         -I<lbr_src> -o iv_jackel_original.dll \
//         jackel_batch_wrapper.cpp LetsBeRational.cpp erf_cody.cpp \
//         rationalcubic.cpp normaldistribution.cpp

#include <cmath>
#include <limits>

// Include LBR header FIRST so its EXPORT macro is already defined
// before we declare our functions (avoids redefinition conflicts)
#include "importexport.h"

// ── our wrapper uses EXPORT_EXTERN_C from LBR's own importexport.h ──────────
// On Windows this expands to:  extern "C" __declspec(dllexport)
// On Linux/macOS this expands to:  extern "C"
// Either way ctypes can find the symbols.

// Jäckel's entry point (defined in LetsBeRational.cpp)
EXPORT_EXTERN_C double normalised_implied_volatility_from_a_transformed_rational_guess(
    double beta, double x, double q
);

// ── Normalisation convention ─────────────────────────────────────────────────
//   Our benchmark:  k = log(K/F),  c = normalised undiscounted call price
//   Jäckel:         x = log(F/K) = -k
//                   beta = c * exp(-0.5*k)   (geometric normalisation)
//                   q = +1 for calls

EXPORT_EXTERN_C void iv_jackel_batch(const double* ks,
                                      const double* cs,
                                      double*       out,
                                      int           n)
{
    for (int i = 0; i < n; ++i) {
        const double k = ks[i];
        const double c = cs[i];

        if (!(c > 0.0) || c != c || k != k) {   // guard NaN / non-positive
            out[i] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }

        const double beta = c * std::exp(-0.5 * k);
        out[i] = normalised_implied_volatility_from_a_transformed_rational_guess(
                     beta, -k, 1.0);
    }
}

EXPORT_EXTERN_C double iv_jackel_scalar(double k, double c)
{
    const double beta = c * std::exp(-0.5 * k);
    return normalised_implied_volatility_from_a_transformed_rational_guess(
               beta, -k, 1.0);
}
