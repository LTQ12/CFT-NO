"""M1: parametric complexity of the shape -> scattering map in three coordinate systems.
One-parameter shape family q(t) = q0 + t*delta, t in [-T, T] (O(1) relative to the wavelength at high k).
At Chebyshev nodes in t we solve exactly and record
  R : raw far field at fixed directions and raw density at fixed polar angles,
  E1: phase-removed envelope v e^{-ik d.x}/k at fixed polar angles (fixed coordinates),
  E2: chart coefficients in charts attached to the moving shadow points (co-moving Fock coordinates).
Parametric bandwidth = number of Chebyshev modes in t needed to reach relative accuracy 1e-6 (and 1e-3)."""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from bie import *; from charts import *
from g2_data import N_LIT, N_SH
import charts as _charts; _charts.SMOOTH_GRADING = True
from numpy.polynomial import chebyshev as C

def cheb_coeffs(vals):
    """Chebyshev coefficients from values at first-kind nodes x_j = cos(pi(j+0.5)/n) (DCT-II)."""
    n = vals.shape[0]
    j = np.arange(n); kk = np.arange(n)
    M = np.cos(np.pi * np.outer(kk, j + 0.5) / n) * (2.0 / n)
    c = M @ vals; c[0] /= 2
    return c

def bandwidth(vals, tol):
    c = cheb_coeffs(vals)
    e = np.sqrt((abs(c) ** 2).sum(1))                       # energy per Chebyshev degree
    tail = np.sqrt(np.cumsum((e ** 2)[::-1])[::-1])         # tail norm from degree m on
    ok = np.nonzero(tail / tail[0] < tol)[0]
    return int(ok[0]) if len(ok) else len(e)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "mode2"
    a0 = np.array([0.0, 0.03, -0.012, 0.008, 0.0]); b0 = np.array([0.0, 0.015, 0.01, -0.006, 0.0])
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03; n_t = 129
    ts = T * np.cos(np.pi * (np.arange(n_t) + 0.5) / n_t)
    alpha0 = 0.3
    alpha = alpha0; d = np.array([np.cos(alpha), np.sin(alpha)])
    xh_dirs = np.array([alpha + np.pi, alpha + np.pi / 2, alpha + 2.0, alpha])     # back, side, oblique, forward
    xh = np.array([np.cos(xh_dirs), np.sin(xh_dirs)])
    out = {}
    for k in [int(x) for x in os.environ.get("KS", "32,64,128,256,512").split(",")]:
        R_ff, R_dn, E1, E2 = [], [], [], []
        NQ = int(2.2 * k * 1.6 / np.pi * 2 + 48)                 # enough for the widest chart (<= 1.6)
        t0 = time.time()
        cache = {}
        for t in ts:
            if which == "rotate":
                alpha = alpha0 + t; d = np.array([np.cos(alpha), np.sin(alpha)])
                xh = np.array([np.cos(xh_dirs + t), np.sin(xh_dirs + t)])       # observation directions rotate with d
            a, b = a0.copy(), b0.copy()
            if which == "mode2":
                a[1] += t
            elif which == "dilation":
                a, b = a0 * (1 + t), b0 * (1 + t)
            r0 = 1.0 + (t if which == "dilation" else 0.0)
            cv = star_curve(a, b, r0)
            g0 = cv.nodes(512); N = int(5.5 * k * g0["sp"].max()) // 2 * 2
            geo = cv.nodes(N)
            if which == "rotate":
                if "A" not in cache:
                    from scipy.linalg import lu_factor, lu_solve
                    mats = assemble(geo, k, need=("S", "Dp"))
                    cache["A"] = lu_factor(0.5 * np.eye(N) + mats["Dp"] - 1j * k * mats["S"]); del mats
                x_ = geo["x"]; n_ = geo["n"]; ui = np.exp(1j * k * (d @ x_))
                from scipy.linalg import lu_solve
                v = lu_solve(cache["A"], 1j * k * (d @ n_) * ui - 1j * k * ui)
            else:
                v = solve_density(geo, k, d[:, None], formulation="cfie")[:, 0]
            R_ff.append(far_field(geo, k, v[:, None], xh)[:, 0])
            # fixed polar angles: lit (theta = alpha+pi), near the two shadow points, mid shadow
            th = np.array([alpha0 + np.pi, alpha0 + np.pi / 2, alpha0 - np.pi / 2, alpha0 + 0.4 * np.pi, alpha0])
            vth = trig_eval(v, th)                            # polar angle = parameter t for star curves
            xth = cv.z(th)
            R_dn.append(vth / k)
            E1.append(vth * np.exp(-1j * k * (d @ xth)) / k)
            ag = ArcGeom(cv, Nf=2 ** 14)
            charts, info = build_charts(ag, d, k, N_LIT, N_SH, 1.0, "full1")
            cf = []
            for (aa, bb, kind) in charts:
                M = 8; nq = NQ                                   # fixed per k: no jumps in the projection as t varies
                s, w, xi = gl_nodes(aa, bb, nq); tt, x, _, _ = ag.point(s)
                vex = trig_eval(v, tt) / k
                B = C.chebvander(xi, M - 1) * np.exp(1j * k * phases(ag, d, s, kind, info)[0])[:, None]
                sw = np.sqrt(w); cf.append(np.linalg.lstsq(B * sw[:, None], vex * sw, rcond=None)[0])
            E2.append(np.concatenate(cf))
        R_ff, R_dn, E1, E2 = map(np.array, (R_ff, R_dn, E1, E2))
        rec = {}
        for name, F in [("raw_farfield", R_ff), ("raw_density_fixed_angle", R_dn), ("envelope_fixed_angle", E1),
                        ("envelope_fixed_angle_lit_only", E1[:, [0, 3]]), ("envelope_fixed_angle_shadow_pts", E1[:, [1, 2]]),
                        ("comoving_chart_coeffs", E2)]:
            rec[name] = {"modes_1e-3": bandwidth(F, 1e-3), "modes_1e-6": bandwidth(F, 1e-6)}
        out[f"k{k}"] = rec
        if os.environ.get("SAVE_COEFFS"):                     # energy per Chebyshev degree, for the decay-rate (Bernstein ellipse) check
            en = {}
            for name, F in [("raw_farfield", R_ff), ("raw_density_fixed_angle", R_dn), ("envelope_fixed_angle", E1),
                            ("envelope_fixed_angle_lit_only", E1[:, [0, 3]]), ("envelope_fixed_angle_shadow_pts", E1[:, [1, 2]]),
                            ("comoving_chart_coeffs", E2)]:
                c = cheb_coeffs(F); en[name] = np.sqrt((abs(c) ** 2).sum(1))
            np.savez(f"results/m1_coeffs_{which}_T{T:g}_k{k}.npz", **en)
        print(which, k, f"{time.time() - t0:.0f}s", {n: (r["modes_1e-3"], r["modes_1e-6"]) for n, r in rec.items()}, flush=True)
        json.dump(dict(T=T, n_t=n_t, family=which, results=out), open(f"results/m1_parametric_{which}_T{T:g}{os.environ.get('OUTTAG', '')}.json", "w"), indent=0)
