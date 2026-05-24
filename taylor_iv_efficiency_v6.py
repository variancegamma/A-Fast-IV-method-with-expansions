#!/usr/bin/env python
"""
taylor_iv_efficiency.py  —  Final consolidated benchmark  (optimised build)

Seed architecture (unchanged)
------------------------------
  ATM  (|k| < 0.01) : Taylor4 ATM inverse
  Mild-OTM  0.01 ≤ |k| ≤ 0.50 : logit/P1 algebraic seed (no exp calls)
  Deep OTM  |k| > 0.50 : ratio-corrected quadratic seed
  Final RootSolver  Polish : Householder-4 everywhere 

Usage
-----
  python taylor_iv_efficiency.py
  python taylor_iv_efficiency.py --repeats 5000 --runs 12
  python taylor_iv_efficiency.py --threads 1 2 4 8
  python taylor_iv_efficiency.py --final-threads 4

Requirements: numpy scipy numba
"""

import argparse
import math
import time

import numpy as np
from scipy.stats import norm

import numba
from numba import njit, prange

SQRT_2PI    = 2.5066282746310005024
INV_PI      = 0.3183098861837906715
# OPT 3: module-level constants — avoids recomputation inside njit functions
INV_SQRT2   = 0.7071067811865475244   # 1 / sqrt(2)
INV_SQRT2PI = 0.3989422804014327      # 1 / sqrt(2·π)

A1          = 0.3989422804014326779   # phi(0) = 1/sqrt(2pi) exact
A1SQ8       = 1.2732395447351626862   # 8·a1² = 4/pi exact
TWO_A1      = 0.7978845608028653559   # 2·a1  = sqrt(2/pi) exact
A3          = -0.06647824564993730    # -a1/6 = -1/(6*sqrt(2pi)) for P3


# ── normal distribution helpers ───────────────────────────────────────────────

@njit(fastmath=True, cache=True)
def normcdf_erf(x):
    # OPT 3: use precomputed module constant instead of recomputing sqrt(2)
    return 0.5 * (1.0 + math.erf(x * INV_SQRT2))

@njit(fastmath=True, cache=True)
def normpdf(x):
    # OPT 3: use precomputed module constant instead of recomputing sqrt(2π)
    return INV_SQRT2PI * math.exp(-0.5 * x * x)

@njit(fastmath=True, cache=True)
def normcdf_inv(p):
    """Acklam rational inverse-normal, max error ~4.5e-9."""
    if p <= 0.0: return -38.0
    if p >= 1.0: return  38.0
    a0=-3.969683028665376e+01; a1= 2.209460984245205e+02
    a2=-2.759285104469687e+02; a3= 1.383577518672690e+02
    a4=-3.066479806614716e+01; a5= 2.506628277459239e+00
    b0=-5.447609879822406e+01; b1= 1.615858368580409e+02
    b2=-1.556989798598866e+02; b3= 6.680131188771972e+01
    b4=-1.328068155288572e+01
    c0=-7.784894002430293e-03; c1=-3.223964580411365e-01
    c2=-2.400758277161838e+00; c3=-2.549732539343734e+00
    c4= 4.374664141464968e+00; c5= 2.938163982698783e+00
    d0= 7.784695709041462e-03; d1= 3.224671290700398e-01
    d2= 2.445134137142996e+00; d3= 3.754408661907416e+00
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) / \
               ((((d0*q+d1)*q+d2)*q+d3)*q+1.0)
    elif p <= 0.97575:
        q = p - 0.5; r = q * q
        return (q*(((((a0*r+a1)*r+a2)*r+a3)*r+a4)*r+a5)) / \
                   (((((b0*r+b1)*r+b2)*r+b3)*r+b4)*r+1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) / \
                ((((d0*q+d1)*q+d2)*q+d3)*q+1.0)




# ── ATM seeds ─────────────────────────────────────────────────────────────────

@njit(fastmath=True, cache=True)
def taylor2_atm(c_tv):
    """v0 = s*(1 + s²/24)"""
    s = SQRT_2PI * c_tv; s2 = s * s
    return max(s * (1.0 + s2 / 24.0), 1e-10)

@njit(fastmath=True, cache=True)
def taylor3_atm(c_tv):
    """v0 = s*(1 + s²/24 + 7s⁴/1920)"""
    s = SQRT_2PI * c_tv; s2 = s * s
    return max(s * (1.0 + s2 * (1.0/24.0 + s2 * 7.0/1920.0)), 1e-10)

@njit(fastmath=True, cache=True)
def taylor4_atm(c_tv):
    """v0 = s*(1 + s²/24 + 7s⁴/1920 + 127s⁶/322560)"""
    s = SQRT_2PI * c_tv; s2 = s * s
    return max(s * (1.0 + s2 * (1.0/24.0
                   + s2 * (7.0/1920.0
                   + s2 *  127.0/322560.0))), 1e-10)


# ── OTM seed variants ─────────────────────────────────────────────────────────

@njit(fastmath=True, cache=True)
def seed_jackel(k, c):
    """Taylor4 ATM + Jäckel quadratic OTM (c ≈ Φ(d₁))."""
    # OPT 5: single exp(k) for ITM; OTM skips it entirely
    if k < 0.0:
        exp_k     = math.exp(k)
        intrinsic = max(1.0 - exp_k, 0.0)
        cc        = max(c - 1.0 + exp_k, 1e-14)
    else:
        intrinsic = 0.0
        cc        = c
    c_tv  = max(c - intrinsic, 1e-14)
    s_atm = taylor4_atm(c_tv)
    kk    = abs(k)
    if kk < 0.01:
        return s_atm
    z  = normcdf_inv(min(cc, 1.0 - 1e-12))
    disc = z*z + 2.0*kk
    if disc <= 0.0: return s_atm
    v_q = z + math.sqrt(disc)
    return v_q if v_q > s_atm else s_atm


@njit(fastmath=True, cache=True)
def seed_sr(k, c):
    """Taylor4 ATM + Stefanica-Radoičić OTM (no Φ⁻¹) + Jäckel fallback."""
    exp_k = math.exp(k)
    intrinsic = max(1.0 - exp_k, 0.0) if k < 0.0 else 0.0
    c_tv  = max(c - intrinsic, 1e-14)
    s_atm = taylor4_atm(c_tv)
    if abs(k) < 0.01:
        return s_atm
    kk = abs(k)
    # OPT 5: exp(|k|) via division — avoids a second math.exp() call
    ek = exp_k if k >= 0.0 else 1.0 / exp_k
    cc = max(c - 1.0 + exp_k, 1e-14) if k < 0.0 else c
    c_adj = cc + 0.5 * (ek - 1.0)
    h     = SQRT_2PI * c_adj / (1.0 + ek)
    disc  = h*h - kk*kk*INV_PI
    if disc >= 0.0:
        v_sr = h + math.sqrt(disc)
        return v_sr if v_sr > s_atm else s_atm
    z    = normcdf_inv(min(cc, 1.0 - 1e-12))
    disc2 = z*z + 2.0*kk
    if disc2 <= 0.0: return s_atm
    v_q  = z + math.sqrt(disc2)
    return v_q if v_q > s_atm else s_atm


@njit(fastmath=True, cache=True)
def seed_ours_no_sr(k, c):
    """Regime-split seed: Taylor4 ATM + ratio-corrected quadratic OTM."""
    exp_k = math.exp(k)
    intrinsic = max(1.0 - exp_k, 0.0) if k < 0.0 else 0.0
    c_tv  = max(c - intrinsic, 1e-14)
    s_atm = taylor4_atm(c_tv)
    if abs(k) < 0.01:
        return s_atm
    kk = abs(k)
    cc = max(c - 1.0 + exp_k, 1e-14) if k < 0.0 else c
    # OPT 5: 1/exp_k avoids second exp() for ITM put-call-parity price
    cs = (1.0 / exp_k) * cc if k < 0.0 else c
    z    = normcdf_inv(min(cs, 1.0 - 1e-12))
    disc = z * z + 2.0 * kk
    if disc <= 0.0:
        return s_atm
    vq = z + math.sqrt(disc)
    alpha = vq * vq / (kk + 0.5 * vq * vq)
    if alpha > 0.0 and alpha < 1.0:
        p_corr = cs / alpha
        if p_corr > 0.0 and p_corr < 1.0 - 1e-12:
            zq    = normcdf_inv(p_corr)
            disc2 = zq * zq + 2.0 * kk
            if disc2 > 0.0:
                vq1 = zq + math.sqrt(disc2)
                return vq1 if vq1 > s_atm else s_atm
    return vq if vq > s_atm else s_atm


@njit(fastmath=True, cache=True)
def seed_sr_ratio(k, c):
    """Taylor4 ATM + SR hybrid + ratio-corrected quadratic for deep OTM."""
    # OPT 5: single exp(k); exp(|k|) recovered via division for ITM
    exp_k = math.exp(k)
    intrinsic = max(1.0 - exp_k, 0.0) if k < 0.0 else 0.0
    c_tv  = max(c - intrinsic, 1e-14)
    s_atm = taylor4_atm(c_tv)
    if abs(k) < 0.01:
        return s_atm
    kk = abs(k)
    ek = exp_k if k >= 0.0 else 1.0 / exp_k
    cc = max(c - 1.0 + exp_k, 1e-14) if k < 0.0 else c
    c_adj = cc + 0.5*(ek - 1.0)
    h     = SQRT_2PI * c_adj / (1.0 + ek)
    disc  = h*h - kk*kk*INV_PI
    if disc >= 0.0:
        v_sr = h + math.sqrt(disc)
        return v_sr if v_sr > s_atm else s_atm
    cs = math.exp(-k) * cc if k < 0.0 else c
    z  = normcdf_inv(min(cs, 1.0 - 1e-12))
    disc2 = z*z + 2.0*kk
    if disc2 <= 0.0: return s_atm
    vq = z + math.sqrt(disc2)
    alpha = vq*vq / (kk + 0.5*vq*vq)
    if alpha > 0.0 and alpha < 1.0:
        p_corr = cs / alpha
        if p_corr > 0.0 and p_corr < 1.0 - 1e-12:
            zq    = normcdf_inv(p_corr)
            disc3 = zq*zq + 2.0*kk
            if disc3 > 0.0:
                vq1 = zq + math.sqrt(disc3)
                return vq1 if vq1 > s_atm else s_atm
    return vq if vq > s_atm else s_atm


@njit(fastmath=True, cache=True)
def seed_sr_gm(k, c):
    """SR hybrid + geometric-mean deep-OTM seed."""
    exp_k = math.exp(k)
    intrinsic = max(1.0 - exp_k, 0.0) if k < 0.0 else 0.0
    c_tv  = max(c - intrinsic, 1e-14)
    s_atm = taylor4_atm(c_tv)
    if abs(k) < 0.01:
        return s_atm
    kk = abs(k)
    # OPT 5: exp(|k|) via division — avoids a second math.exp() call
    ek = exp_k if k >= 0.0 else 1.0 / exp_k
    cc = max(c - 1.0 + exp_k, 1e-14) if k < 0.0 else c
    c_adj = cc + 0.5 * (ek - 1.0)
    h     = SQRT_2PI * c_adj / (1.0 + ek)
    disc  = h*h - kk*kk*INV_PI
    if disc >= 0.0:
        v_sr = h + math.sqrt(disc)
        return v_sr if v_sr > s_atm else s_atm
    z     = normcdf_inv(min(cc, 1.0 - 1e-12))
    disc2 = z*z + 2.0*kk
    v_j   = z + math.sqrt(disc2) if disc2 > 0.0 else s_atm
    v_s = math.sqrt(2.0 * kk)
    for _ in range(2):
        inner = cc * kk * SQRT_2PI / v_s
        if inner <= 0.0 or inner >= 1.0: break
        lg = math.log(inner)
        if lg >= 0.0: break
        v_s = kk / math.sqrt(-2.0 * lg)
    v_gm = math.sqrt(v_j * v_s) if v_j > 0.0 and v_s > 0.0 else v_j
    return v_gm if v_gm > s_atm else s_atm


# ── Logit/P1 clean seed ───────────────────────────────────────────────────────

@njit(fastmath=True, cache=True)
def seed_logit_clean(k, c):
    """
    Logit/P1 clean algebraic seed + ratio-corrected quadratic for deep OTM.
    v2 changes:
      - Exact slope a1 = phi(0) = 1/sqrt(2pi) replaces fitted coefficient.
      - P3 cubic one-step Newton correction for mild-OTM (0.01 <= |k| <= 0.50):
          Phi(x) ~ 1/2 + a1*x + a3*x^3,  a3 = -a1/6
          T3(v)  = eps*kk^3/v^3 + 1.5*(2+eps)*kk^2/v + 0.75*eps*kk*v + (2+eps)/8*v^3
          vP3    = vP1 - F3(vP1) / F3'(vP1)
        Reduces mild-OTM mean seed error ~6x vs P1.
        On Windows (slow erf) saves ~2-3 ns/IV net; neutral on Linux.
    """
    kk = abs(k)
    if k < 0.0:
        exp_k     = math.exp(k)
        inv_ek    = 1.0 / exp_k
        intrinsic = max(1.0 - exp_k, 0.0)
    else:
        intrinsic = 0.0

    c_tv  = max(c - intrinsic, 1e-14)
    s_atm = taylor4_atm(c_tv)
    if kk < 0.01:
        return s_atm

    # ── mild-OTM: logit/P1 + P3 cubic Newton correction ──────────────────────
    if kk <= 0.50:
        eps = kk * (1.0 + kk * (0.5 + kk * (1.0/6.0 + kk / 24.0)))
        if k < 0.0:
            cc = max(c - 1.0 + exp_k, 1e-14)
            cs = inv_ek * cc
        else:
            cs = c
        den = 2.0 + eps
        h2  = 2.0 * cs + eps
        N   = h2 * h2 - A1SQ8 * kk * eps * den
        vp1 = (h2 + math.sqrt(N)) / (TWO_A1 * den) if N > 0.0 else h2 / (TWO_A1 * den)
        if math.isfinite(vp1) and 1e-10 < vp1 < 5.0:
            # ── P3 cubic one-step Newton correction ──────────────────────────
            v   = vp1
            v2  = v  * v;  v3 = v2 * v
            kk2 = kk * kk; kk3 = kk2 * kk
            h   = cs + 0.5 * eps           # h = cseed + eps/2
            A_  = A1 * eps * kk
            B_  = A1 * den * 0.5
            T3  = (eps * kk3 / v3
                   + 1.5 * den * kk2 / v
                   + 0.75 * eps * kk * v
                   + den / 8.0 * v3)
            F3  = A_ / v + B_ * v + A3 * T3 - h
            dT3 = (-3.0 * eps * kk3 / (v3 * v)
                   - 1.5 * den * kk2 / v2
                   + 0.75 * eps * kk
                   + 3.0 * den / 8.0 * v2)
            dF3 = -A_ / v2 + B_ + A3 * dT3
            if abs(dF3) > 1e-20:
                vp3 = v - F3 / dF3
                if math.isfinite(vp3) and 1e-10 < vp3 < 5.0:
                    v = vp3
            return v if v > s_atm else s_atm

    # ── deep OTM |k| > 0.50: ratio-corrected quadratic (vq1) ────────────────
    if k < 0.0:
        cc = max(c - 1.0 + exp_k, 1e-14)
        cs = inv_ek * cc
    else:
        cs = c
    z     = normcdf_inv(min(cs, 1.0 - 1e-12))
    disc  = z * z + 2.0 * kk
    if disc <= 0.0:
        return s_atm
    vq = z + math.sqrt(disc)
    alpha = vq * vq / (kk + 0.5 * vq * vq)
    if 0.0 < alpha < 1.0:
        p_corr = cs / alpha
        if 0.0 < p_corr < 1.0 - 1e-12:
            zq    = normcdf_inv(p_corr)
            disc2 = zq * zq + 2.0 * kk
            if disc2 > 0.0:
                vq1 = zq + math.sqrt(disc2)
                return vq1 if vq1 > s_atm else s_atm
    return vq if vq > s_atm else s_atm


# ── Halley polisher ───────────────────────────────────────────────────────────

@njit(fastmath=True, cache=True)
def halley_iv(k, c, v):
    # OPT 1: exp(k) hoisted outside the loop.
    # OPT 6: cap reduced 8→6 — sufficient for all regime-split seeds,
    #         tighter bound helps compiler schedule the loop.
    ek = math.exp(k)
    for _ in range(8):
        d1 = -k/v + 0.5*v; d2 = d1 - v
        f  = normcdf_erf(d1) - ek * normcdf_erf(d2) - c
        if abs(f) < 1e-14: break
        vg = normpdf(d1)
        if vg <= 1e-15: break
        vom = vg*d1*d2/v; den = 2.0*vg*vg - f*vom
        vn  = (v - 2.0*f*vg/den) if abs(den) > 1e-20 else (v - f/vg)
        if not math.isfinite(vn) or vn <= 1e-10 or vn > 10.0: break
        v = vn
    return v


# ── serial batch kernels ──────────────────────────────────────────────────────

@njit(fastmath=True, parallel=False, cache=True)
def batch_jackel_serial(ks, cs, out):
    for i in range(ks.size):
        v0 = seed_jackel(ks[i], cs[i])
        out[i] = halley_iv(ks[i], cs[i], max(min(v0, 4.9), 1e-10))

@njit(fastmath=True, parallel=False, cache=True)
def batch_sr_serial(ks, cs, out):
    for i in range(ks.size):
        v0 = seed_sr(ks[i], cs[i])
        out[i] = halley_iv(ks[i], cs[i], max(min(v0, 4.9), 1e-10))

@njit(fastmath=True, parallel=False, cache=True)
def batch_sr_ratio_serial(ks, cs, out):
    for i in range(ks.size):
        v0 = seed_sr_ratio(ks[i], cs[i])
        out[i] = halley_iv(ks[i], cs[i], max(min(v0, 4.9), 1e-10))

@njit(fastmath=True, parallel=False, cache=True)
def batch_sr_gm_serial(ks, cs, out):
    for i in range(ks.size):
        v0 = seed_sr_gm(ks[i], cs[i])
        out[i] = halley_iv(ks[i], cs[i], max(min(v0, 4.9), 1e-10))

@njit(fastmath=True, parallel=False, cache=True)
def batch_logit_clean_serial(ks, cs, out):
    for i in range(ks.size):
        v0 = seed_logit_clean(ks[i], cs[i])
        out[i] = halley_iv(ks[i], cs[i], max(min(v0, 4.9), 1e-10))


# ── parallel batch kernels ────────────────────────────────────────────────────

@njit(fastmath=True, parallel=True, cache=True)
def batch_jackel_parallel(ks, cs, out):
    for i in prange(ks.size):
        v0 = seed_jackel(ks[i], cs[i])
        out[i] = halley_iv(ks[i], cs[i], max(min(v0, 4.9), 1e-10))

@njit(fastmath=True, parallel=True, cache=True)
def batch_sr_parallel(ks, cs, out):
    for i in prange(ks.size):
        v0 = seed_sr(ks[i], cs[i])
        out[i] = halley_iv(ks[i], cs[i], max(min(v0, 4.9), 1e-10))

@njit(fastmath=True, parallel=True, cache=True)
def batch_sr_ratio_parallel(ks, cs, out):
    for i in prange(ks.size):
        v0 = seed_sr_ratio(ks[i], cs[i])
        out[i] = halley_iv(ks[i], cs[i], max(min(v0, 4.9), 1e-10))

@njit(fastmath=True, parallel=True, cache=True)
def batch_sr_gm_parallel(ks, cs, out):
    for i in prange(ks.size):
        v0 = seed_sr_gm(ks[i], cs[i])
        out[i] = halley_iv(ks[i], cs[i], max(min(v0, 4.9), 1e-10))

@njit(fastmath=True, parallel=True, cache=True)
def batch_logit_clean_parallel(ks, cs, out):
    for i in prange(ks.size):
        v0 = seed_logit_clean(ks[i], cs[i])
        out[i] = halley_iv(ks[i], cs[i], max(min(v0, 4.9), 1e-10))



def bs_call_price(F: float, K: float, T: float, sigma: float) -> float:
    """Undiscounted forward measure call price."""
    if T <= 0.0 or sigma <= 0.0:
        return max(F - K, 0.0)
    vol = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol) / vol
    d2 = d1 - vol
    return F * norm.cdf(d1) - K * norm.cdf(d2)

@njit(fastmath=True)
def householder4_iv(k, c, v):
    """Householder-4 (quartic) polisher for the mild-OTM logit band."""
    ek = math.exp(k)
    for _ in range(6):
        d1  = -k/v + 0.5*v; d2 = d1 - v
        f   = normcdf_erf(d1) - ek*normcdf_erf(d2) - c
        if abs(f) < 1e-14: break
        vg  = normpdf(d1)
        if vg <= 1e-15: break
        p   = d1 * d2
        r   = f / vg                              # Newton correction
        alp = p / v                               # f'' / f'
        bet = (p*p - (d1*d1 + d2*d2) - p) / (v*v)  # f''' / f'
        num =  3.0 * r * (2.0 - r * alp)
        den = -6.0 + r * (6.0 * alp - r * bet)
        vn  = v + (num / den if abs(den) > 1e-20 else -r)
        if not math.isfinite(vn) or vn <= 1e-10 or vn > 5.0: break
        v = vn
    return v

@njit(fastmath=True, parallel=False)
def batch_nosr_logit_hh4_serial(ks, cs, out):
    """LogitNorm P1 clean seed + Householder-4 polish (mild-OTM), Halley elsewhere."""
    for i in range(ks.size):
        v0 = seed_logit_clean(ks[i], cs[i])
        out[i] = polish_nosr_logit_hh4(ks[i], cs[i], max(min(v0, 4.9), 1e-10))

@njit(fastmath=True)
def polish_nosr_logit_hh4(k, c, v):
    """Dispatch: mild-OTM logit band → HH4 (quartic), else → Halley (cubic)."""
    kk = abs(k)
    if 0.01 <= kk <= 0.50:
        return householder4_iv(k, c, v)
    return halley_iv(k, c, v)


@njit(fastmath=True, parallel=False, cache=True)
def batch_logit_hh4all_serial(ks, cs, out):
    """LogitNorm P1 clean seed + Householder-4 polish in all regimes."""
    for i in range(ks.size):
        v0 = seed_logit_clean(ks[i], cs[i])
        out[i] = householder4_iv(ks[i], cs[i], max(min(v0, 4.9), 1e-10))

@njit(fastmath=True, parallel=True, cache=True)
def batch_logit_hh4all_parallel(ks, cs, out):
    """LogitNorm P1 clean seed + Householder-4 polish in all regimes (parallel)."""
    for i in prange(ks.size):
        v0 = seed_logit_clean(ks[i], cs[i])
        out[i] = householder4_iv(ks[i], cs[i], max(min(v0, 4.9), 1e-10))
# ── grid builder ──────────────────────────────────────────────────────────────

def build_grid(repeats=5000):
    vols   = np.r_[0.01, np.arange(0.05, 2.0001, 0.05)]
    deltas = np.array([0.05, 0.20, 0.30, 0.45, 0.55, 0.70, 0.80, 0.95])
    ks, vs, cs = [], [], []
    for v in vols:
        for D in deltas:
            k  = v * (0.5*v - norm.ppf(D))
            d1 = -k/v + 0.5*v; d2 = d1 - v
            c  = norm.cdf(d1) - np.exp(k)*norm.cdf(d2)
            ks.append(k); vs.append(v); cs.append(float(c))
    ks = np.array(ks, dtype=np.float64)
    vs = np.array(vs, dtype=np.float64)
    cs = np.array(cs, dtype=np.float64)
    return (np.ascontiguousarray(np.tile(ks, repeats)),
            np.ascontiguousarray(np.tile(cs, repeats)),
            np.tile(vs, repeats),
            ks, cs, vs)


# ── analysis helpers ──────────────────────────────────────────────────────────

def print_atm_seed_table():
    print("\n" + "="*78)
    print("A. ATM seed accuracy before Halley polishing")
    print("="*78)
    vols = [0.01, 0.05, 0.20, 0.40, 0.90, 1.20, 1.50, 2.00]
    print(f"{'v exact':>7} {'seed':<12} {'v0':>20} {'abs err':>14} {'rel err':>14}")
    print("-"*78)
    for v in vols:
        c = 2.0 * norm.cdf(v / 2.0) - 1.0
        s = math.sqrt(2.0 * math.pi) * c
        seeds = [
            ("B-S s",   s),
            ("Taylor2", s * (1.0 + s*s/24.0)),
            ("Taylor3", s * (1.0 + s*s/24.0 + s**4 * 7.0/1920.0)),
            ("Taylor4", s * (1.0 + s*s/24.0 + s**4 * 7.0/1920.0 + s**6 * 127.0/322560.0)),
        ]
        for name, vv in seeds:
            ae = abs(vv - v); re = ae / v
            print(f"{v:7.2f} {name:<12} {vv:20.15f} {ae:14.3e} {re:14.3e}")
        print("-"*78)


def count_halley_iters(ks_b, cs_b, seed_fn):
    iters = []
    for i in range(len(ks_b)):
        k, c = float(ks_b[i]), float(cs_b[i])
        vv   = float(max(min(float(seed_fn(k, c)), 4.9), 1e-10))
        for n_it in range(1, 15):
            d1 = -k/vv + 0.5*vv; d2 = d1 - vv
            b  = 0.5*(1+math.erf(d1/math.sqrt(2))) \
               - math.exp(k)*0.5*(1+math.erf(d2/math.sqrt(2)))
            f  = b - c
            if abs(f) < 1e-14: break
            vg = math.exp(-0.5*d1*d1) / math.sqrt(2*math.pi)
            if vg <= 1e-15: break
            vom = vg*d1*d2/vv; den = 2*vg*vg - f*vom
            vn  = (vv - 2*f*vg/den) if abs(den) > 1e-20 else (vv - f/vg)
            if not math.isfinite(vn) or vn <= 1e-10 or vn > 5.0: break
            vv  = vn
        iters.append(n_it)
    arr = np.array(iters)
    return arr.mean(), arr.max(), np.bincount(arr, minlength=10).tolist()


def bench_kernel(label, fn, ks, cs, true_v, out, runs=12):
    fn(ks, cs, out)   # warm-up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(ks, cs, out)
        times.append(time.perf_counter() - t0)
    ns  = np.array(times) * 1e9 / ks.size
    err = np.abs(out - true_v).max()
    print(f"  {label:<46} best={ns.min():7.1f}  mean={ns.mean():7.1f}  "
          f"median={np.median(ns):7.1f}  std={ns.std(ddof=1):5.1f}  "
          f"max_err={err:.2e}")
    return ns, err

#%%
# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats",       type=int,   default=5000)
    parser.add_argument("--runs",          type=int,   default=5)
    parser.add_argument("--threads",       nargs="+",  type=int,
                        default=[1, 2])
                                 # 4, 8, 12])
    parser.add_argument("--final-threads", type=int,   default=4)
    args = parser.parse_args()

    print("Black-Scholes IV regime-split benchmark  —  optimised build")
    print(f"Numba {numba.__version__}  |  "
          f"Available threads: {numba.get_num_threads()}")

    # ── A. ATM accuracy table ─────────────────────────────────────────────────
    print_atm_seed_table()

    # ── B. Build grid + iteration counts ─────────────────────────────────────
    print("\n" + "="*78)
    print("B. Grid and Halley iteration distribution")
    print("="*78)

    ks, cs, true_v, ks_b, cs_b, vs_b = build_grid(args.repeats)
    out = np.empty_like(ks)
    print(f"  Grid: {ks.size:,} IVs  (328 base × {args.repeats} repeats)")

    print(f"\n  {'Seed':<24}  {'mean':>6}  {'max':>4}  distribution (iters 1..9+)")
    print("  " + "-"*68)
    for lbl, sfn in [("Brenner-Subrahmanyam + Halley (baseline)",       seed_jackel),
                     ("SR hybrid",                seed_sr),
                     ("SR + ratio-corrected",     seed_sr_ratio),
                     ("SR + GM deep-OTM",         seed_sr_gm),
                     ("logit_clean + ratio-corr", seed_logit_clean)]:
        mn, mx, dist = count_halley_iters(ks_b, cs_b, sfn)
        print(f"  {lbl:<24}  {mn:6.4f}  {mx:4d}  {dist[1:]}")

    # ── compile ───────────────────────────────────────────────────────────────
    print("\n  Compiling kernels ...", end="", flush=True)
    for fn in (batch_jackel_serial, batch_sr_serial,
               batch_sr_ratio_serial, batch_sr_gm_serial,
               batch_nosr_logit_hh4_serial,
               batch_logit_hh4all_serial,
               batch_jackel_parallel, batch_sr_parallel,
               batch_sr_ratio_parallel, batch_sr_gm_parallel,
               batch_logit_clean_parallel,
               batch_logit_hh4all_parallel):
        fn(ks[:2000], cs[:2000], out[:2000])
    print(" done")

    # ── C. Serial benchmark ───────────────────────────────────────────────────
    print("\n" + "="*78)
    print("C. Serial numba benchmark")
    print("="*78)
    print(f"  {'label':<46} {'best':>7}  {'mean':>7}  "
          f"{'median':>7}  {'std':>5}  max_err")
    bench_kernel("Brenner-Subrahmanyam + Halley (baseline)", batch_jackel_serial,
                 ks, cs, true_v, out, args.runs)
    bench_kernel("SR hybrid",         batch_sr_serial,
                 ks, cs, true_v, out, args.runs)
    bench_kernel("SR + ratio-corrected OTM", batch_sr_ratio_serial,
                 ks, cs, true_v, out, args.runs)
    bench_kernel("SR + GM deep-OTM",  batch_sr_gm_serial,
                 ks, cs, true_v, out, args.runs)
    bench_kernel("logit_clean + ratio-corrected OTM", batch_logit_clean_serial,
                 ks, cs, true_v, out, args.runs),
    bench_kernel("logit_clean + ratio-corrected_HH4 OTM", batch_nosr_logit_hh4_serial,
                 ks, cs, true_v, out, args.runs)
    bench_kernel("logit_clean + HH4 all regimes",    batch_logit_hh4all_serial,
                 ks, cs, true_v, out, args.runs)

    # ── D. Parallel thread sweep ──────────────────────────────────────────────
    max_threads = numba.get_num_threads()
    threads_to_test = [t for t in args.threads if t <= max_threads]

    if threads_to_test:
        print("\n" + "="*78)
        print("D. Parallel thread sweep")
        print("="*78)
        print(f"  {'label':<46} {'best':>7}  {'mean':>7}  "
              f"{'median':>7}  {'std':>5}  max_err")
        for nt in threads_to_test:
            numba.set_num_threads(nt)
            bench_kernel(f"Brenner-Subrahmanyam + Halley  {nt}t", batch_jackel_parallel,
                         ks, cs, true_v, out, args.runs)
            bench_kernel(f"SR      {nt}t", batch_sr_parallel,
                         ks, cs, true_v, out, args.runs)
            bench_kernel(f"SR+ratio {nt}t", batch_sr_ratio_parallel,
                         ks, cs, true_v, out, args.runs)
            bench_kernel(f"SR+GM   {nt}t", batch_sr_gm_parallel,
                         ks, cs, true_v, out, args.runs)
            bench_kernel(f"logit_clean {nt}t", batch_logit_clean_parallel,
                         ks, cs, true_v, out, args.runs)
            bench_kernel(f"logit_hh4all {nt}t", batch_logit_hh4all_parallel,
                         ks, cs, true_v, out, args.runs)

    # ── E. Final repeated benchmark ───────────────────────────────────────────
    print("\n" + "="*78)
    print("E. Final repeated benchmark")
    print("="*78)

    ft = min(args.final_threads, max_threads)
    numba.set_num_threads(ft)

    print(f"  {ft} thread(s),  {args.runs} runs")
    print(f"  {'label':<46} {'best':>7}  {'mean':>7}  "
          f"{'median':>7}  {'std':>5}  max_err")

    fn_j  = batch_jackel_parallel  if ft > 1 else batch_jackel_serial
    fn_s  = batch_sr_parallel      if ft > 1 else batch_sr_serial
    fn_r  = batch_sr_ratio_parallel if ft > 1 else batch_sr_ratio_serial
    fn_g  = batch_sr_gm_parallel   if ft > 1 else batch_sr_gm_serial
    fn_lc = batch_logit_clean_parallel if ft > 1 else batch_logit_clean_serial
    fn_h4 = batch_logit_hh4all_parallel if ft > 1 else batch_logit_hh4all_serial

    ns_j,  _ = bench_kernel("Brenner-Subrahmanyam + Halley (baseline)", fn_j,
                             ks, cs, true_v, out, args.runs)
    ns_s,  _ = bench_kernel("SR hybrid",         fn_s,
                             ks, cs, true_v, out, args.runs)
    ns_r,  _ = bench_kernel("SR + ratio-corrected OTM", fn_r,
                             ks, cs, true_v, out, args.runs)
    ns_g,  _ = bench_kernel("SR + GM deep-OTM",  fn_g,
                             ks, cs, true_v, out, args.runs)
    ns_lc, _ = bench_kernel("logit_clean + ratio-corrected OTM", fn_lc,
                             ks, cs, true_v, out, args.runs)
    ns_h4, _ = bench_kernel("logit_clean + HH4 all regimes",     fn_h4,
                             ks, cs, true_v, out, args.runs)

    print(f"\n  Summary (best-run):")
    print(f"    Brenner-Subrahmanyam + Halley baseline          : {ns_j.min():.1f} ns/IV")
    print(f"    SR hybrid                : {ns_s.min():.1f} ns/IV  "
          f"({ns_j.min()/ns_s.min():.3f}x vs Jäckel)")
    print(f"    SR + ratio-corrected     : {ns_r.min():.1f} ns/IV  "
          f"({ns_j.min()/ns_r.min():.3f}x vs Jäckel)")
    print(f"    SR + GM deep-OTM         : {ns_g.min():.1f} ns/IV  "
          f"({ns_j.min()/ns_g.min():.3f}x vs Jäckel)")
    print(f"    logit_clean + ratio-corr : {ns_lc.min():.1f} ns/IV  "
          f"({ns_j.min()/ns_lc.min():.3f}x vs Jäckel)")
    print(f"    logit_clean + HH4-all    : {ns_h4.min():.1f} ns/IV  "
          f"({ns_j.min()/ns_h4.min():.3f}x vs Jäckel)")


if __name__ == "__main__":
    main()
