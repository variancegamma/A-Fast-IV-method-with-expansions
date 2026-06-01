#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "iv_regime_final_v2.c"
//"iv_regime_final.c"
//"iv_regime_three_band_tail_cs_0p01.c"
//"iv_regime_three_band.c"



#ifdef _WIN32
#include <windows.h>

static double seconds_now(void) {
    LARGE_INTEGER freq;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
}

#else
#include <time.h>

static double seconds_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
#endif
// static double seconds_now(void) {
//     struct timespec ts;
//     clock_gettime(CLOCK_MONOTONIC, &ts);
//     return (double)ts.tv_sec + 1e-9*(double)ts.tv_nsec;
// }

int main(void) {
    const int nvol = 41, ndel = 8, repeats = 5000, nbase = 328;
    const int n = nbase * repeats, runs = 10;
    double *ks = (double*)malloc((size_t)n * sizeof(double));
    double *cs = (double*)malloc((size_t)n * sizeof(double));
    double *true_v = (double*)malloc((size_t)n * sizeof(double));
    double *out = (double*)malloc((size_t)n * sizeof(double));
    double vols[41], deltas[8] = {0.05,0.20,0.30,0.45,0.55,0.70,0.80,0.95};
    vols[0] = 0.01;
    for (int i=1; i<nvol; ++i) vols[i] = 0.05*i;
    int idx=0;
    for (int r=0; r<repeats; ++r) {
        for (int i=0; i<nvol; ++i) {
            double v = vols[i];
            for (int j=0; j<ndel; ++j) {
                double D = deltas[j];
                double k = v*(0.5*v - normcdf_inv(D));
                double d1 = -k/v + 0.5*v;
                double d2 = d1 - v;
                double c = normcdf(d1) - exp(k)*normcdf(d2);
                ks[idx]=k; cs[idx]=c; true_v[idx]=v; ++idx;
            }
        }
    }
    iv_regime_batch(ks, cs, out, n); // warm-up/cache
    double ns[10];
    for (int run=0; run<runs; ++run) {
        double t0=seconds_now();
        iv_regime_batch(ks, cs, out, n);
        double t1=seconds_now();
        ns[run] = 1e9*(t1-t0)/n;
    }
    double maxerr=0.0;
    for (int i=0; i<n; ++i) {
        double e = fabs(out[i]-true_v[i]);
        if (e>maxerr) maxerr=e;
    }
    double best=ns[0], mean=0.0;
    for (int i=0; i<runs; ++i) { if (ns[i]<best) best=ns[i]; mean += ns[i]; }
    mean/=runs;
    for (int i=0;i<runs-1;i++) for (int j=i+1;j<runs;j++) if(ns[j]<ns[i]) {double tmp=ns[i]; ns[i]=ns[j]; ns[j]=tmp;}
    double median=0.5*(ns[4]+ns[5]);
    printf("C no-SR + Logit/Taylor P1 mild-OTM benchmark\n");
    printf("Grid size: %d IVs (328 x %d)\n", n, repeats);
    printf("best=%10.1f ns/IV   mean=%10.1f   median=%10.1f   max_err=%.3e\n", best, mean, median, maxerr);
    free(ks); free(cs); free(true_v); free(out);
    return 0;
}
