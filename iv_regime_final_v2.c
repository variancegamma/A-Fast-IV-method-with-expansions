/* iv_regime_hh4_all_p7.c — experimental P7 Taylor-CDF seed branch */
#include <math.h>

#if defined(_WIN32) || defined(_WIN64)
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

#ifndef P7_CUT
#define P7_CUT 1.347
#endif

/* avg(vP3,vP7) is applied when AVG_LO < |k| <= P7_CUT         */
/* Theory: beyond k=0.81 P3 undershoots and P7 overshoots;     */
/* their average brackets the truth and reduces seed error ~7%  */
#ifndef AVG_LO
#define AVG_LO 0.81
#endif
#ifndef AVG_HI
#define AVG_HI 1.155
#endif

#define SQRT_2PI   2.5066282746310005024
#define INV_SQRT2  0.7071067811865475244
#define INV_SQRT2PI 0.3989422804014326779
#define A1         0.3989422804014326779
#define A1SQ8      1.2732395447351626862
#define TWO_A1     0.7978845608028653559
#define A3        -0.0664903800669054463  /* -A1/6 */
#define A5         0.00997355701003581695 /* A1/40 */
#define A7        -0.0011873282154804544  /* -A1/336 */

/* P3 cubic CDF approximation and derivative */
static inline double poly3(double x) {
    double x2 = x*x;
    return 0.5 + x * (A1 + x2 * A3);
}
static inline double dpoly3(double x) {
    double x2 = x*x;
    return A1 + x2 * 3.0*A3;
}

static inline double normcdf(double x) { return 0.5 * (1.0 + erf(x * INV_SQRT2)); }
static inline double normpdf(double x) { return INV_SQRT2PI * exp(-0.5 * x * x); }

static double normcdf_inv(double p) {
    if (p <= 0.0) return -38.0;
    if (p >= 1.0) return  38.0;
    static const double a[] = {-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00};
    if (p < 0.02425) {
        double q = sqrt(-2.0 * log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= 0.97575) {
        double q = p - 0.5, r = q * q;
        return (q*(((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])) /
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        double q = sqrt(-2.0 * log(1.0 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}

static inline double taylor4_atm(double c_tv) {
    double s = SQRT_2PI * c_tv, s2 = s * s;
    return s * (1.0 + s2 * (1.0/24.0 + s2 * (7.0/1920.0 + s2 * 127.0/322560.0)));
}

static inline double poly7(double x) {
    double x2 = x*x;
    return 0.5 + x * (A1 + x2 * (A3 + x2 * (A5 + x2 * A7)));
}
static inline double dpoly7(double x) {
    double x2 = x*x;
    return A1 + x2 * (3.0*A3 + x2 * (5.0*A5 + x2 * 7.0*A7));
}

static inline double seed_ratio_corr_otm(double k, double c,
                                          double kk, double s_atm,
                                          double exp_k, double inv_ek) {
    double cc = (k < 0.0) ? fmax(c - 1.0 + exp_k, 1e-14) : c;
    double cs = (k < 0.0) ? inv_ek * cc : c;
    double z  = normcdf_inv(fmin(cs, 1.0 - 1e-12));
    double disc = z * z + 2.0 * kk;
    if (disc <= 0.0) return s_atm;
    double vq = z + sqrt(disc);
    double alpha = vq * vq / (kk + 0.5 * vq * vq);
    if (alpha > 0.0 && alpha < 1.0) {
        double pc = cs / alpha;
        if (pc > 0.0 && pc < 1.0 - 1e-12) {
            double zq = normcdf_inv(pc);
            double disc2 = zq * zq + 2.0 * kk;
            if (disc2 > 0.0) {
                double vq1 = zq + sqrt(disc2);
                return (vq1 > s_atm) ? vq1 : s_atm;
            }
        }
    }
    return (vq > s_atm) ? vq : s_atm;
}

static inline double vp1_clean_eps(double kk, double cs, double eps, double s_atm) {
    double den = 2.0 + eps;
    double h2  = 2.0 * cs + eps;
    double N   = h2 * h2 - A1SQ8 * kk * eps * den;
    double vp1 = (N > 0.0) ? (h2 + sqrt(N)) / (TWO_A1 * den)
                            :  h2             / (TWO_A1 * den);
    return (isfinite(vp1) && vp1 > 1e-10 && vp1 < 5.0) ? ((vp1 > s_atm) ? vp1 : s_atm) : s_atm;
}

static inline double seed_p7_newton(double kk, double cs, double E,
                                     double vp1, double s_atm) {
    double v = vp1;
    if (!isfinite(v) || v <= 1e-10 || v > 5.0) return s_atm;

    double d1    = -kk/v + 0.5*v;
    double d2    = d1 - v;
    double invv2 = 1.0/(v*v);
    double d1p   = kk*invv2 + 0.5;
    double d2p   = kk*invv2 - 0.5;

    /* vP7: one Newton step on P7 surrogate */
    double F7  = poly7(d1) - E * poly7(d2) - cs;
    double dF7 = dpoly7(d1)*d1p - E*dpoly7(d2)*d2p;
    double vp7 = v;
    if (fabs(dF7) > 1e-20) {
        double vn = v - F7/dF7;
        if (isfinite(vn) && vn > 1e-10 && vn < 5.0) vp7 = vn;
    }

    /* For kk > AVG_LO: P7 overshoots while P3 undershoots —        */
    /* Three sub-regimes beyond AVG_LO:                              */
    /*   AVG_LO < kk <= AVG_HI : avg(vP3,vP7) — true bracket        */
    /*   AVG_HI < kk <= P7_CUT : vP3 alone — crosses v at k~1.134,  */
    /*     eta_P3 < 3% vs eta_avg up to 8.67%                        */
    if (kk > AVG_LO) {
        double F3  = poly3(d1) - E * poly3(d2) - cs;
        double dF3 = dpoly3(d1)*d1p - E*dpoly3(d2)*d2p;
        double vp3 = v;
        if (fabs(dF3) > 1e-20) {
            double vn = v - F3/dF3;
            if (isfinite(vn) && vn > 1e-10 && vn < 5.0) vp3 = vn;
        }
        if (kk <= AVG_HI) {
            double vavg = 0.5*(vp3 + vp7);
            return (vavg > s_atm) ? vavg : s_atm;
        }
        /* kk > AVG_HI: vP3 alone is near-exact */
        return (vp3 > s_atm) ? vp3 : s_atm;
    }

    return (vp7 > s_atm) ? vp7 : s_atm;
}

static double regime_seed(double k, double c) {
    double kk = fabs(k);
    double exp_k = 0.0, inv_ek = 0.0, intrinsic = 0.0;
    if (k < 0.0) {
        exp_k = exp(k);
        inv_ek = 1.0 / exp_k;
        intrinsic = fmax(1.0 - exp_k, 0.0);
    }
    double c_tv = fmax(c - intrinsic, 1e-14);
    double s_atm = taylor4_atm(c_tv);
    /* ATM cutoff lowered to 0.001: at v=0.01 this is |k|/v=0.1, genuinely ATM.
     * Points 0.001<=|k|<0.01 are handled by vP7 below, which respects |k|/v
     * through d1,d2 and avoids the Taylor4 low-volatility basis error.       */
    if (kk < 0.001) return s_atm;

    double cs, E;
    if (k < 0.0) {
        double cc = fmax(c - 1.0 + exp_k, 1e-14);
        cs = inv_ek * cc;
        E = inv_ek;           /* exp(kk) */
    } else {
        cs = c;
        E = exp(kk);          /* needed for P7 branch */
    }

    /* Tail override: extreme small OTM-equivalent price with moderate-to-deep
     * moneyness is better initialised by vq1 than by the polynomial seed.
     * Guard: kk > 0.5 prevents misfiring on mild-OTM low-vol points where
     * cs<0.01 reflects low volatility, not deep moneyness.              */
    if (cs < 0.02128 && kk > 0.5) {
        return seed_ratio_corr_otm(k, c, kk, s_atm, exp_k, inv_ek);
    }

    if (kk <= P7_CUT) {
        double eps = kk * (1.0 + kk * (0.5 + kk * (1.0/6.0 + kk * (1.0/24.0))));
        double vp1 = vp1_clean_eps(kk, cs, eps, s_atm);
        return seed_p7_newton(kk, cs, E, vp1, s_atm);
    }
    return seed_ratio_corr_otm(k, c, kk, s_atm, exp_k, inv_ek);
}

static double householder4(double k, double v, double c) {
    const double ek = exp(k);
    for (int i = 0; i < 6; i++) {
        double d1  = -k/v + 0.5*v;
        double d2  = d1 - v;
        double f   = normcdf(d1) - ek * normcdf(d2) - c;
        if (fabs(f) < 1e-14) break;
        double vg  = normpdf(d1);
        if (vg <= 1e-15) break;
        double p   = d1 * d2;
        double r   = f / vg;
        double alp = p / v;
        double bet = (p*p - (d1*d1 + d2*d2) - p) / (v*v);
        double num =  3.0 * r * (2.0 - r * alp);
        double den = -6.0 + r * (6.0 * alp - r * bet);
        double vn  = v + ((fabs(den) > 1e-20) ? num / den : -r);
        if (!isfinite(vn) || vn <= 1e-10 || vn > 5.0) break;
        v = vn;
    }
    return v;
}

EXPORT void iv_regime_batch(const double * __restrict__ ks,
                             const double * __restrict__ cs,
                             double       * __restrict__ out,
                             int n) {
    for (int i = 0; i < n; i++) {
        double v  = regime_seed(ks[i], cs[i]);
        if (v < 1e-10) v = 1e-10;
        if (v > 4.9)   v = 4.9;
        out[i] = householder4(ks[i], v, cs[i]);
    }
}

EXPORT double iv_regime_scalar(double k, double c) {
    double v = regime_seed(k, c);
    if (v < 1e-10) v = 1e-10;
    if (v > 4.9)   v = 4.9;
    return householder4(k, v, c);
}
