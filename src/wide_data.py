"""Training data of the chart-local surrogate, generated on Colab, plus a high-frequency extension.
Splits train / val (stars, k in [32,128]) and train_hi / val_hi (k in [128,512]) are those of the paper model,
same seeds and loops as g2_data.py; train_xhi / val_xhi are new: 200 / 20 stars, one k in [512, 2048]
(log-uniform), 8 incident directions each.
Per trace: features and least-squares chart coefficients exactly as g2_data.make_sample; the density is the
CFIE Nystrom solution at 5 points per wavelength computed on the GPU (gpu_bie), and the trigonometric
interpolation of the density at the chart quadrature nodes (the expensive part of the fit) runs on the GPU
in one product for all 32 charts.
Checks: the regenerated coefficients of the paper splits are compared with values of the original files
(CHECKS below) before a split is saved.
Usage: python wide_data.py train val train_hi val_hi train_xhi val_xhi    -> $WIDE_DIR/g2_<split>.npz"""
import os, sys, time, json, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
WIDE_DIR = os.environ.get("WIDE_DIR", "/content/wide_data" if os.path.isdir("/content") else os.path.join(ROOT, "wide_data"))
os.makedirs(WIDE_DIR, exist_ok=True)
import g2_data as G
from g2_data import chart_features, radial_descriptor, nodes_for, N_LIT, N_SH, M, FB
from charts import ArcGeom, build_charts, phases
from shapes import random_convex_star
from numpy.polynomial import chebyshev as C
import gpu_bie

SPLITS = {"train": (400, 100, 2, 6, 32, 128), "val": (40, 200, 2, 6, 32, 128),
          "train_hi": (300, 150, 1, 8, 128, 512), "val_hi": (30, 250, 1, 8, 128, 512),
          "train_xhi": (200, 160, 1, 8, 512, 2048), "val_xhi": (20, 260, 1, 8, 512, 2048)}

# (sample, chart, coefficient, Re Y, Im Y, ||Y_sample||) from the original g2_<split>.npz of the paper model
CHECKS = {
    "train": [(0, 5, 0, 4.255009099430619e-06, 4.649449501656325e-06, 0.018930827854038497), (0, 20, 1, -2.4890193890998513e-05, 3.1648336914725135e-05, 0.018930827854038497), (2000, 5, 0, -1.0053224241553227e-07, -5.284758832638552e-07, 0.01762962917321106), (2000, 20, 1, -0.0004465698304931177, 1.8798776652445537e-05, 0.01762962917321106), (4799, 5, 0, 2.741132946643168e-06, 2.6965186390763087e-06, 0.014729627615854718), (4799, 20, 1, -0.0007492508519058613, 7.301719630576549e-05, 0.014729627615854718)],
    "val": [(0, 5, 0, -1.2526618445346122e-06, -2.2489326401303944e-06, 0.039965245283873234), (0, 20, 1, -0.0008425886001159645, 0.0001850024386790142, 0.039965245283873234), (479, 5, 0, 2.1096781714158566e-05, -3.2758513157293277e-07, 0.02115062171067627), (479, 20, 1, -0.0005617707863279624, 6.752994306418733e-05, 0.02115062171067627)],
    "train_hi": [(0, 5, 0, -8.61397058083406e-08, 5.087419568872165e-08, 0.006803347495749832), (0, 20, 1, -6.588437513267997e-05, 1.0947515435976306e-06, 0.006803347495749832), (2399, 5, 0, -1.5937114751744997e-07, 3.222617690363498e-07, 0.01786668866737724), (2399, 20, 1, -0.0007676262133417698, 4.870428744620049e-05, 0.01786668866737724)],
    "val_hi": [(0, 5, 0, 2.3516713490417433e-07, -2.744268905201701e-07, 0.019091346304981927), (0, 20, 1, 0.0006868774307952414, -3.325346527396324e-05, 0.019091346304981927), (239, 5, 0, -3.419983545730223e-07, 9.92700553550083e-07, 0.011410004271277642), (239, 20, 1, -0.0003421335132330522, 6.994172571854215e-06, 0.011410004271277642)],
}

_GL = {}
def leggauss(n):
    if n not in _GL:
        _GL[n] = np.polynomial.legendre.leggauss(n)
    return _GL[n]


class TrigEval:
    """Trigonometric interpolant of nodal values at arbitrary parameters t, on the GPU if available."""
    def __init__(self):
        self.t = None
        try:
            import torch
            if torch.cuda.is_available():
                self.t = torch
        except Exception:
            pass

    def __call__(self, vnodes, t):
        N = len(vnodes); c = np.fft.fft(vnodes) / N; m = np.fft.fftfreq(N, 1.0 / N); c[N // 2] = 0
        if self.t is None:
            out = np.empty(len(t), complex)
            for i in range(0, len(t), 2000):
                out[i:i + 2000] = np.exp(1j * np.outer(t[i:i + 2000], m)) @ c
            return out
        T = self.t; dev = "cuda"
        tt = T.as_tensor(t, dtype=T.float64, device=dev); mm = T.as_tensor(m, dtype=T.float64, device=dev)
        cc = T.as_tensor(c, dtype=T.complex128, device=dev); out = []
        for i in range(0, len(t), 4096):
            ph = tt[i:i + 4096, None] * mm[None, :]
            out.append(T.complex(T.cos(ph), T.sin(ph)) @ cc)
        return T.cat(out).cpu().numpy()


TE = TrigEval()


def fit_correction_fast(ag, vnodes, k, d, charts, info):
    """g2_data.fit_correction with one interpolation and one Fock-base evaluation for all charts."""
    s_in, s_out = info["s_in"], info["s_out"]
    S, Wq, XI, cut = [], [], [], [0]
    for (a, b, kind) in charts:
        nq = int(max(4 * M, 2.2 * k * (b - a) / np.pi * 2 + 3 * M))
        xg, wg = leggauss(nq)
        S.append(0.5 * (b - a) * xg + 0.5 * (a + b)); Wq.append(0.5 * (b - a) * wg); XI.append(xg); cut.append(cut[-1] + nq)
    s_all = np.concatenate(S); t_all, _, _, _ = ag.point(s_all)
    vex_all = TE(vnodes, t_all); vb_all = FB.density(ag, s_all, k, d, s_in, s_out)
    coefs, widths = [], []
    for j, (a, b, kind) in enumerate(charts):
        sl = slice(cut[j], cut[j + 1]); s, w, xi = S[j], Wq[j], XI[j]
        B = C.chebvander(xi, M - 1) * np.exp(1j * k * phases(ag, d, s, kind, info)[0])[:, None]
        sw = np.sqrt(w)
        coefs.append(np.linalg.lstsq(B * sw[:, None], (vex_all[sl] - vb_all[sl]) / k * sw, rcond=None)[0])
        widths.append(b - a)
    return np.array(coefs), np.array(widths)


def make_sample(cv, ag, vnodes, k, alpha):
    d = np.array([np.cos(alpha), np.sin(alpha)])
    charts, info = build_charts(ag, d, k, N_LIT, N_SH, 1.0, "full1")
    if not hasattr(ag, "I_f"):
        FB.density(ag, np.array([0.0]), k, d, info["s_in"], info["s_out"])
    desc = radial_descriptor(ag, alpha)
    X = chart_features(ag, k, d, charts, info, desc)
    Y, W = fit_correction_fast(ag, vnodes, k, d, charts, info)
    return X, Y, W


def density(B, cv, k, alphas):
    geo = cv.nodes(nodes_for(cv, k))
    if B is None:
        from bie import solve_density
        return solve_density(geo, k, np.array([np.cos(alphas), np.sin(alphas)]), formulation="cfie")
    return gpu_bie.cfie_density(B, geo, k, alphas)


def run(split, B):
    path = os.path.join(WIDE_DIR, f"g2_{split}.npz")
    if os.path.exists(path):
        print(split, "already generated, skipped", flush=True); return
    nshape, seed, n_k, n_dir, klo, khi = SPLITS[split]
    rng = np.random.default_rng(seed)
    Xs, Ys, Ws, ks, als = [], [], [], [], []
    t0 = time.time(); ts = tf = 0.0
    for i in range(nshape):
        cv = random_convex_star(rng, name=f"{split}{i}"); ag = ArcGeom(cv)
        for kk in np.exp(rng.uniform(np.log(klo), np.log(khi), n_k)):
            alphas = rng.uniform(0, 2 * np.pi, n_dir)
            t1 = time.time(); V = density(B, cv, kk, alphas); ts += time.time() - t1
            t1 = time.time()
            for j, al in enumerate(alphas):
                X, Y, W = make_sample(cv, ag, V[:, j], kk, al)
                Xs.append(X); Ys.append(Y); Ws.append(W); ks.append(kk); als.append(al)
            tf += time.time() - t1
        if i % 20 == 0 or i == nshape - 1:
            print(f"  {split} shape {i + 1}/{nshape}  traces {len(Xs)}  solve {ts:.0f}s  fit {tf:.0f}s  ({time.time() - t0:.0f}s)", flush=True)
    Y = np.array(Ys)
    for (i, c, m, re, im, nrm) in CHECKS.get(split, []):
        e = abs(Y[i, c, m] - complex(re, im)) / nrm
        print(f"  check {split} sample {i} chart {c}: |dY|/||Y|| = {e:.1e}", flush=True)
        assert e < 1e-4, "regenerated coefficients differ from the paper data"
    np.savez(path, X=np.array(Xs), Y=Y, W=np.array(Ws), k=np.array(ks), alpha=np.array(als))
    print(f"{split} saved: {len(Xs)} traces ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    splits = sys.argv[1:] or list(SPLITS)
    B = None if os.environ.get("LOCAL_TEST") else __import__("bench_data_colab").gpu_backend()
    if B is None and not os.environ.get("LOCAL_TEST"):
        print("WARNING: no GPU, densities on the CPU (slow)", flush=True)
    print("trigonometric interpolation on", "GPU" if TE.t is not None else "CPU", flush=True)
    for sp in splits:
        run(sp, B)
