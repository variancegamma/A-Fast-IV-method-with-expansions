"""
bench_jackel_vs_regime.py

Ready-to-compile benchmark for Peter Jäckel's original LetsBeRational source
against your regime-split IV implementation.

What it does:
1. Downloads the vollib/lets_be_rational source archive unless --lbr-dir is given.
2. Compiles Jäckel's C++ source + jackel_batch_wrapper.cpp into a shared library.
3. Compiles your iv_regime.c into a shared library.
4. Runs the same 328-point Jäckel grid timing benchmark.

Usage, Linux/macOS:
    python bench_jackel_vs_regime.py

Usage, Windows with MinGW-w64 in PATH:
    python bench_jackel_vs_regime.py

Offline usage:
    git clone https://github.com/vollib/lets_be_rational.git
    python bench_jackel_vs_regime.py --lbr-dir path/to/lets_be_rational

Requirements:
    numpy scipy
    g++ and gcc in PATH
    iv_regime.c in the same folder as this script, or pass --regime-c path/to/iv_regime.c
"""

import argparse
import ctypes
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
import math
from numba import njit
from urllib.request import urlretrieve

import numpy as np
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
LBR_ZIP_URL = "https://github.com/vollib/lets_be_rational/archive/refs/heads/master.zip"


def lib_ext():
    s = platform.system()
    if s == "Windows":
        return ".dll"
    if s == "Darwin":
        return ".dylib"
    return ".so"


def run_cmd(cmd, cwd=None):
    print(" ".join(map(str, cmd)))
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if r.returncode != 0:
        print("\n--- STDOUT ---")
        print(r.stdout)
        print("\n--- STDERR ---")
        print(r.stderr)
        raise SystemExit(r.returncode)
    return r


def ensure_lbr_source(lbr_dir=None):
    if lbr_dir is not None:
        p = Path(lbr_dir).resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    build = HERE / "_lbr_build"
    build.mkdir(exist_ok=True)
    dst = build / "lets_be_rational_master.zip"
    root = build / "lets_be_rational-master"

    if not root.exists():
        print(f"Downloading LetsBeRational source from {LBR_ZIP_URL}")
        urlretrieve(LBR_ZIP_URL, dst)
        with zipfile.ZipFile(dst, "r") as zf:
            zf.extractall(build)

    return root


def find_case_insensitive(root, wanted):
    wanted_l = wanted.lower()
    hits = [p for p in Path(root).rglob("*") if p.is_file() and p.name.lower() == wanted_l]
    if not hits:
        raise FileNotFoundError(f"Could not find {wanted} under {root}")
    return hits[0]


def compile_jackel(lbr_root):
    src_names = [
        "LetsBeRational.cpp",
        "erf_cody.cpp",
        "rationalcubic.cpp",
        "normaldistribution.cpp",
    ]
    src_files = [find_case_insensitive(lbr_root, name) for name in src_names]
    src_dir = src_files[0].parent

    wrapper = HERE / "jackel_batch_wrapper.cpp"
    out = HERE / ("iv_jackel_original" + lib_ext())

    flags =["-Ofast", "-march=native", "-ffp-contract=fast","-DNDEBUG"] 
    #["-O3", "-ffast-math", "-march=native", "-DNDEBUG"]
    if platform.system() != "Windows":
        flags += ["-fPIC"]

    cmd = ["g++"] + flags + ["-shared", "-I", str(src_dir), "-o", str(out), str(wrapper)]
    cmd += [str(p) for p in src_files]
    if platform.system() != "Windows":
        cmd += ["-lm"]

    print("\nCompiling original Jäckel LetsBeRational shared library:")
    run_cmd(cmd)
    return out


def compile_regime(regime_c):
    regime_c = Path(regime_c).resolve()
    if not regime_c.exists():

        raise FileNotFoundError(f"Missing regime source: {regime_c}")

    out = HERE / ("iv_regime_hh4_all" + lib_ext())
    flags =["-Ofast", "-march=native", "-mtune=native","-ffp-contract=fast","-fno-math-errno", "-fno-trapping-math", "-fomit-frame-pointer"] 
    #["-O3", "-ffast-math", "-march=native"]
    if platform.system() != "Windows":
        flags += ["-fPIC"]

    cmd = ["gcc"] + flags + ["-shared", "-o", str(out), str(regime_c)]
    if platform.system() != "Windows":
        cmd += ["-lm"]

    print("\nCompiling your regime-split C shared library:")
    run_cmd(cmd)
    return out


def load_batch(path, fn_name):
    lib = ctypes.CDLL(str(path))
    fn = getattr(lib, fn_name)
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    fn.restype = None

    def batch(ks, cs, out):
        fn(
            ks.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(len(ks)),
        )

    return batch

@njit(fastmath=True)
def normcdf_inv(p):
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
    pl, ph = 0.02425, 0.97575
    if pl <= p <= ph:
        q = p - 0.5; r = q*q
        return (q*(((((a0*r+a1)*r+a2)*r+a3)*r+a4)*r+a5) /
                   (((((b0*r+b1)*r+b2)*r+b3)*r+b4)*r+1.0))
    elif p < pl:
        q = math.sqrt(-2.0*math.log(p))
        return (((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) / \
               ((((d0*q+d1)*q+d2)*q+d3)*q+1.0)
    else:
        q = math.sqrt(-2.0*math.log(1.0-p))
        return -(((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) / \
                ((((d0*q+d1)*q+d2)*q+d3)*q+1.0)

@njit(fastmath=True)
def normcdf_erf(x): return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))
@njit(fastmath=True)
def normpdf(x): return math.exp(-0.5*x*x)/math.sqrt(2.0*math.pi)



def build_grid(repeats=5000):
    vols = np.r_[0.01, np.arange(0.05, 2.0001, 0.05)]
    deltas = np.array([0.05,0.20, 0.30, 0.45, 0.55, 0.70, 0.80, 0.95])
    ks, vs, cs = [], [], []

    for v in vols:
        for D in deltas:
            k = v * (0.5 * v - norm.ppf(D))
            d1 = -k / v + 0.5 * v
            d2 = d1 - v
            c = norm.cdf(d1) - np.exp(k) * norm.cdf(d2)
            ks.append(k)
            vs.append(v)
            cs.append(float(c))

    ks = np.asarray(ks, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    cs = np.asarray(cs, dtype=np.float64)

    return (
        np.ascontiguousarray(np.tile(ks, repeats)),
        np.ascontiguousarray(np.tile(cs, repeats)),
        np.tile(vs, repeats),
    )


def bench(label, fn, ks, cs, true_v, runs=10):
    out = np.empty_like(ks)

    # warm-up
    fn(ks, cs, out)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(ks, cs, out)
        times.append(time.perf_counter() - t0)

    dt = min(times)
    err = np.abs(out - true_v)
    print(f"  {label:<42} {1e9*dt/len(ks):9.1f} ns/IV   max_err={err.max():.3e}")
    return dt, err.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lbr-dir", default="C:/Users/aheki/Downloads/_lbr_build", help="Existing lets_be_rational source tree")
    ap.add_argument("--regime-c", default=str(HERE / "iv_regime_hh4_all.c"), help="Path to your iv_regime.c")
    ap.add_argument("--repeats", type=int, default=5000)
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    print("=" * 78)
    print(" Jäckel original LetsBeRational vs regime-split IV benchmark")
    print(f" Platform: {platform.system()} {platform.machine()}")
    print("=" * 78)

    lbr_root = ensure_lbr_source(args.lbr_dir)
    jackel_lib = compile_jackel(lbr_root)
    regime_lib = compile_regime(args.regime_c)

    jackel_batch = load_batch(jackel_lib, "iv_jackel_batch")
    regime_batch = load_batch(regime_lib, "iv_regime_batch")

    print(f"\nBuilding grid: 328 points × {args.repeats} repeats")
    ks, cs, tv = build_grid(args.repeats)

    print("\nTiming:")
    print(f"  {'Method':<42} {'ns/IV':>9}   max_err")
    print("  " + "─" * 67)
    t_j, e_j = bench("Original Jäckel LetsBeRational", jackel_batch, ks, cs, tv, args.runs)
    t_r, e_r = bench("Regime-split + Halley", regime_batch, ks, cs, tv, args.runs)
    

    print("  " + "─" * 67)
    print(f"  Speedup Paper's method vs Jäckel: {t_j/t_r:.3f}×  ({100*(1-t_r/t_j):.1f}% faster)")
    print("\nDone.")
    ks_test = np.array([np.log(16.48721271/10.0)])
    cs_test = np.array([0.9989800933915837])
    out_test = np.empty(1, dtype=np.float64)
    jackel_batch(ks_test, cs_test, out_test)
    print("Small test Jackel:", out_test[0])

if __name__ == "__main__":
    main()
