"""G2 data: chart-local features -> Fock-base correction coefficients.

Surrogate density:  v_hat = v_Fock (PO + Fock corrections, analytic)
                         + k * sum_charts sum_m c_m T_m(xi) exp(ik phase_chart)
Charts: 32 per sample (9 lit + 7 shadow per shadow-boundary point), graded at the Fock scale,
lit charts carry the incident phase, shadow charts the creeping phase of their own point.
Features depend only on geometry, k and d; targets are the least-squares coefficients."""
import sys, time, json, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from bie import *; from charts import *; from fock import FockBase
from shapes import random_convex_star
from numpy.polynomial import chebyshev as C

import os as _os
import charts as _charts
SUFFIX = ""
if _os.environ.get("SMOOTH") == "1":
    _charts.SMOOTH_GRADING = True
    SUFFIX = "_smooth"
M = 8
N_LIT, N_SH = 9, 7
_D = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
FB = FockBase(next(p for p in (_D + "/results/fock_table.npz", _D + "/fock_table.npz") if __import__("os").path.exists(p)))


def radial_descriptor(ag, alpha, nmodes=6):
    """Fourier coefficients (modes 1..nmodes) of log r(theta) in the frame where d = (1, 0)."""
    th = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    s = np.linspace(0, ag.L, 4096, endpoint=False)
    _, x, _, _ = ag.point(s)
    ang = np.mod(np.arctan2(x[1], x[0]) - alpha, 2 * np.pi)
    o = np.argsort(ang)
    lr = np.interp(th, ang[o], np.log(np.hypot(x[0], x[1]))[o], period=2 * np.pi)
    c = np.fft.rfft(lr) / len(th)
    return np.concatenate([[c[0].real], c[1:nmodes + 1].real, c[1:nmodes + 1].imag])


def chart_features(ag, k, d, charts, info, desc):
    """One feature row per chart."""
    I = lambda q: np.interp(np.mod(q, ag.L), ag.s_f, ag.I_f)
    Itot = ag.I_f[-1]; c = (k / 2) ** (1 / 3)
    s_in, s_out = info["s_in"], info["s_out"]
    rows = []
    for ci, (a, b, kind) in enumerate(charts):
        # which shadow point owns this chart, and on which side
        if kind == "sh_in" or (kind == "lit" and ci >= 2 * N_SH + N_LIT):
            sb, side = s_in, 1.0
        else:
            sb, side = s_out, -1.0
        shadow = kind != "lit"
        def xi_of(q):
            if shadow:
                return c * ((I(q) - I(sb)) % Itot if side > 0 else (I(sb) - I(q)) % Itot)
            return -c * ((I(sb) - I(q)) % Itot if side > 0 else (I(q) - I(sb)) % Itot)
        xa, xb_ = xi_of(np.array([a + 1e-12]))[0], xi_of(np.array([b - 1e-12]))[0]
        q = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(np.pi * (np.arange(5) + 0.5) / 5)
        _, x, n, kap = ag.point(q)
        _, _, nn, kk = ag.point(np.array([a, 0.5 * (a + b), b, sb]))
        dn = d @ nn[:, :3]
        rows.append(np.concatenate([
            [float(shadow), side, np.arcsinh(xa / 2), np.arcsinh(xb_ / 2), np.log((b - a) * k ** (1 / 3)),
             np.log(k / 64), np.log(kk[3])],
            np.log(kap), dn, desc]))
    return np.array(rows)


def fit_correction(ag, vnodes, k, d, charts, info):
    s_in, s_out = info["s_in"], info["s_out"]
    coefs, widths = [], []
    for (a, b, kind) in charts:
        nq = int(max(4 * M, 2.2 * k * (b - a) / np.pi * 2 + 3 * M))
        s, w, xi = gl_nodes(a, b, nq); t, x, _, _ = ag.point(s)
        vex = trig_eval(vnodes, t); vb = FB.density(ag, s, k, d, s_in, s_out)
        B = C.chebvander(xi, M - 1) * np.exp(1j * k * phases(ag, d, s, kind, info)[0])[:, None]
        sw = np.sqrt(w)
        coefs.append(np.linalg.lstsq(B * sw[:, None], (vex - vb) / k * sw, rcond=None)[0])
        widths.append(b - a)
    return np.array(coefs), np.array(widths)


def make_sample(cv, ag, vnodes, k, alpha):
    d = np.array([np.cos(alpha), np.sin(alpha)])
    charts, info = build_charts(ag, d, k, N_LIT, N_SH, 1.0, "full1")
    if not hasattr(ag, "I_f"):
        FB.density(ag, np.array([0.0]), k, d, info["s_in"], info["s_out"])   # builds ag.I_f
    desc = radial_descriptor(ag, alpha)
    X = chart_features(ag, k, d, charts, info, desc)
    Y, W = fit_correction(ag, vnodes, k, d, charts, info) if vnodes is not None else (None, None)
    return X, Y, W, charts, info


def nodes_for(cv, k):
    g0 = cv.nodes(512)
    return int(5 * k * g0["sp"].max()) // 2 * 2


if __name__ == "__main__":
    split = sys.argv[1]
    if split in ("train_rich", "val_rich"):
        # local-geometry coverage: support-function bodies (other seeds than the test set) + ellipses
        from shapes import random_support_body
        from bie import ellipse
        n_sup, n_ell, seed = (240, 60, 400) if split == "train_rich" else (24, 6, 500)
        rng = np.random.default_rng(seed)
        Xs, Ys, Ws, meta, Vs = [], [], [], [], {}
        t0 = time.time()
        shapes = [random_support_body(rng, name=f"{split}_sup{i}") for i in range(n_sup)]
        for i in range(n_ell):
            asp = rng.uniform(1.2, 2.5); b = 1.0 / np.sqrt(asp)
            shapes.append(ellipse(b * asp, b, name=f"{split}_ell{i}"))
        for i, cv in enumerate(shapes):
            ag = ArcGeom(cv)
            kk = float(np.exp(rng.uniform(np.log(32), np.log(512))))
            N = nodes_for(cv, kk); geo = cv.nodes(N)
            alphas = rng.uniform(0, 2 * np.pi, 8)
            V = solve_density(geo, kk, np.array([np.cos(alphas), np.sin(alphas)]), formulation="cfie")
            for j, al in enumerate(alphas):
                X, Y, W, _, _ = make_sample(cv, ag, V[:, j], kk, al)
                Xs.append(X); Ys.append(Y); Ws.append(W)
                meta.append(dict(shape=i, name=cv.name, k=kk, alpha=float(al), params=cv.params))
                Vs[f"s{len(Xs) - 1}"] = V[:, j].astype(np.complex64)
            if i % 20 == 0:
                print(split, i, f"{time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(f"results/g2_{split}{SUFFIX}.npz", X=np.array(Xs), Y=np.array(Ys), W=np.array(Ws))
        json.dump(meta, open(f"results/g2_{split}{SUFFIX}_meta.json", "w"))
        np.savez_compressed(f"results/g2_{split}{SUFFIX}_densities.npz", **Vs)
    elif split in ("train", "val", "train_hi", "val_hi"):
        nshape, seed, n_k, n_dir = {"train": (400, 100, 2, 6), "val": (40, 200, 2, 6),
                                    "train_hi": (300, 150, 1, 8), "val_hi": (30, 250, 1, 8)}[split]
        klo, khi = (32, 128) if split in ("train", "val") else (128, 512)
        rng = np.random.default_rng(seed)
        Xs, Ys, Ws, meta = [], [], [], []
        Vs = {}
        t0 = time.time()
        for i in range(nshape):
            cv = random_convex_star(rng, name=f"{split}{i}"); ag = ArcGeom(cv)
            for kk in np.exp(rng.uniform(np.log(klo), np.log(khi), n_k)):
                N = nodes_for(cv, kk); geo = cv.nodes(N)
                alphas = rng.uniform(0, 2 * np.pi, n_dir)
                V = solve_density(geo, kk, np.array([np.cos(alphas), np.sin(alphas)]))
                for j, al in enumerate(alphas):
                    X, Y, W, _, _ = make_sample(cv, ag, V[:, j], kk, al)
                    Xs.append(X); Ys.append(Y); Ws.append(W)
                    if SUFFIX:
                        Vs[f"s{len(Xs) - 1}"] = V[:, j].astype(np.complex64)
                    meta.append(dict(shape=i, k=float(kk), alpha=float(al), params=cv.params))
            if i % 20 == 0:
                print(split, i, f"{time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(f"results/g2_{split}{SUFFIX}.npz", X=np.array(Xs), Y=np.array(Ys), W=np.array(Ws))
        json.dump(meta, open(f"results/g2_{split}{SUFFIX}_meta.json", "w"))
        if SUFFIX:
            np.savez_compressed(f"results/g2_{split}{SUFFIX}_densities.npz", **Vs)
    else:   # test: new stars + ellipses, k up to 512; store exact densities for evaluation
        from bie import ellipse
        rng = np.random.default_rng(300)
        shapes = [random_convex_star(rng, name=f"test{i}") for i in range(24)]
        shapes += [ellipse(a, b, name=f"testell{i}") for i, (a, b) in enumerate([(1.0, 0.7), (1.1, 0.8), (0.9, 0.65), (1.0, 0.8)])]
        out = {}; meta = []
        t0 = time.time()
        for i, cv in enumerate(shapes):
            ag = ArcGeom(cv)
            alphas = rng.uniform(0, 2 * np.pi, 3)
            for kk in [32, 64, 128, 256, 512]:
                N = nodes_for(cv, kk); geo = cv.nodes(N)
                mats = assemble(geo, kk)
                V = solve_density(geo, kk, np.array([np.cos(alphas), np.sin(alphas)]), mats)
                del mats
                for j, al in enumerate(alphas):
                    key = f"{cv.name}_k{kk}_d{j}"
                    out[key] = V[:, j]
                    meta.append(dict(key=key, name=cv.name, k=kk, alpha=float(al), params=cv.params, N=N))
            print("test", i, f"{time.time() - t0:.0f}s", flush=True)
            np.savez_compressed("results/g2_test_densities.npz", **out)
            json.dump(meta, open("results/g2_test_meta.json", "w"))


GROUPS = [(0, 7, None), (7, 14, None), (14, 23, 7), (23, 32, 0)]   # (start, end, chart before the first one)


def add_context(X):
    """Append the features of the inward and outward neighbour chart (same shadow point, ordered from
    the shadow point outward). X: (..., 32, F) -> (..., 32, 3F). Missing neighbours are zero."""
    Z = np.zeros_like(X)
    inner, outer = Z.copy(), Z.copy()
    for a, b, before in GROUPS:
        outer[..., a:b - 1, :] = X[..., a + 1:b, :]
        inner[..., a + 1:b, :] = X[..., a:b - 1, :]
        if before is not None:
            inner[..., a, :] = X[..., before, :]
    return np.concatenate([X, inner, outer], -1)
