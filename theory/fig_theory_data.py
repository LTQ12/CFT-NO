"""Data for the theory figure (Section 3): rotation family of M1 (incident angle 0.3 + t, t in [-0.4, 0.4]).
For k = 32, 128, 512, one Nystrom CFIE matrix (5.5 points per wavelength) is factorised and solved for all
incident directions. Recorded at Chebyshev points in t:
  h  : phase-removed trace v e^{-ik d.x}/k at the fixed polar angle theta0 = 0.3 + pi/2, which a shadow point crosses
       near t = 0 (Theorem 3.4(b));
  kink: the physical-optics current 2i min(d.n, 0) at that point, the k -> infinity limit of h at the layer scale;
  w  : phase-removed trace at the geometric co-moving points Y_t(sigma) with fixed xi = lambda sigma in XI,
       divided by e_k(sigma) = k^{-1/3}(1 + |xi|) (Corollary 3.7).
Output: fig_theory_data.npz next to this script."""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from bie import star_curve, assemble
from charts import ArcGeom, shadow_points, trig_eval
from scipy.linalg import lu_factor, lu_solve

a0 = np.array([0.0, 0.03, -0.012, 0.008, 0.0]); b0 = np.array([0.0, 0.015, 0.01, -0.006, 0.0])
cv = star_curve(a0, b0, 1.0); ag = ArcGeom(cv, Nf=2 ** 15)
alpha0, T, n_t = 0.3, 0.4, 257
ts = T * np.cos(np.pi * (np.arange(n_t) + 0.5) / n_t)[::-1]
theta0 = alpha0 + np.pi / 2
XI = np.array([-2.0, 0.0, 1.0])
KS = [int(x) for x in os.environ.get("KS", "32,64,128,256,512").split(",")]

# geometric Fock coordinate around the shadow point near theta0, for every t
sg = np.linspace(0, ag.L, 2 ** 15, endpoint=False)
_, _, _, kap_g = ag.point(sg)
def sigma_points(d, xi_over_lam):
    s_in, s_out = shadow_points(ag, d)
    tin, tout = ag.t_of_s(np.array([s_in, s_out]))
    dist = lambda a: abs((a - theta0 + np.pi) % (2 * np.pi) - np.pi)
    if dist(tin) < dist(tout):
        sp, sgn = s_in, +1          # shadow for s > s_in
    else:
        sp, sgn = s_out, -1         # shadow for s < s_out
    u = np.linspace(-0.9, 0.9, 4001)                      # arclength offsets into the shadow (u > 0)
    _, _, _, kp = ag.point(sp + sgn * u)
    w = kp ** (2 / 3)
    sig = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(u))]); sig -= np.interp(0.0, u, sig)
    from scipy.interpolate import CubicSpline
    uu = CubicSpline(sig, u)(xi_over_lam)
    return ag.t_of_s(sp + sgn * uu)

out = dict(ts=ts, XI=XI, ks=np.array(KS))
xf = cv.z(np.array([theta0])); d1 = cv.dz(np.array([theta0])); sp1 = np.hypot(*d1)
nf = np.array([d1[1], -d1[0]]) / sp1
D = np.array([np.cos(alpha0 + ts), np.sin(alpha0 + ts)])
out["kink"] = 2j * np.minimum((D.T @ nf)[:, 0], 0.0)
for k in KS:
    t0 = time.time()
    g0 = cv.nodes(512); N = int(5.5 * k * g0["sp"].max()) // 2 * 2
    geo = cv.nodes(N)
    mats = assemble(geo, k, need=("S", "Dp"))
    A = lu_factor(0.5 * np.eye(N) + mats["Dp"] - 1j * k * mats["S"]); del mats
    x_, n_ = geo["x"], geo["n"]
    ui = np.exp(1j * k * (D.T @ x_))                      # (n_t, N)
    rhs = (1j * k * (D.T @ n_) * ui - 1j * k * ui).T
    V = lu_solve(A, rhs)                                  # (N, n_t)
    h = trig_eval(V, np.array([theta0]))[0] * np.exp(-1j * k * (D.T @ xf)[:, 0]) / k
    lam = (k / 2) ** (1 / 3); W = np.zeros((len(XI), n_t), complex)
    for j, t in enumerate(ts):
        th = sigma_points(D[:, j], XI / lam)
        vv = trig_eval(V[:, j:j + 1], th)[:, 0]; xx = cv.z(th)
        W[:, j] = vv * np.exp(-1j * k * (D[:, j] @ xx)) / k / (k ** (-1 / 3) * (1 + abs(XI)))
    out[f"h_{k}"] = h; out[f"w_{k}"] = W
    print(f"k={k} N={N} done ({time.time() - t0:.0f}s)", flush=True)
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_theory_data.npz"), **out)
