"""Surrogate forward map: (shape, k, d) -> Neumann trace -> far field.
v_hat = Fock base (analytic) + chart correction predicted by the chart-local MLP."""
import sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from charts import *; from g2_data import make_sample, M, FB
from mlp import MLP
from numpy.polynomial import chebyshev as C


class Surrogate:
    def __init__(self, path, base="fock"):
        self.base = base
        self.net, z = MLP.load(path)
        self.xm, self.xs, self.p, self.ys = z["xm"], z["xs"], float(z["p"]), z["ys"]
        self.flags = str(z["variant"]).split("_") if "variant" in z.files else []
        import charts as _charts
        _charts.SMOOTH_GRADING = "smooth" in self.flags      # chart layout must match the one used in training
        if "inf" in self.flags:                              # base built from the k -> infinity Fock table
            zt = np.load(next(p for p in [__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), d, "fock_table_inf.npz") for d in ("results", "")] if __import__("os").path.exists(p)))
            FB.F, FB.G, FB._sp = zt["F"], zt["G"], {}

    def prep(self, X):
        if "dimless" in self.flags:
            import dimless; dimless.DOD = "dod" in self.flags
            from dimless import to_dimless, gate_context
            if "local" in self.flags:
                X = X[..., :-13]
            self._raw = X.copy(); X = to_dimless(X)
            if "ctx" in self.flags:
                from g2_data import add_context
                X = gate_context(add_context(X), self._raw)
            return X
        if "ctx" in self.flags:
            from g2_data import add_context
            if "local" in self.flags:
                X = X[..., :-13]
            X = add_context(X)
        return X

    def coefs(self, cv, ag, k, alpha):
        X, _, _, charts, info = make_sample(cv, ag, None, k, alpha)
        X = self.prep(X)
        Yn = self.net.forward(((X - self.xm) / self.xs).astype(np.float32)).astype(np.float64)
        W = np.array([b - a for a, b, _ in charts])
        if "dimless" in self.flags:
            from dimless import scale_factor
            Y = Yn * self.ys / scale_factor(k, W, self._raw, self.p)[:, None]
        else:
            Y = Yn * self.ys / (k ** self.p * np.sqrt(W)[:, None])
        return charts, info, Y[:, :M] + 1j * Y[:, M:]

    @staticmethod
    def density(ag, s, k, d, charts, info, coef, base=True, corr=True, base_kind="fock"):
        parts = ("po", "fock") if base_kind == "fock" else ("po",)
        out = FB.density(ag, s, k, d, info["s_in"], info["s_out"], parts) if base else np.zeros(len(s), complex)
        if corr:
            for (a, b, kind), cf in zip(charts, coef):
                sm = (s - a) % ag.L
                sel = sm < (b - a)
                if sel.any():
                    ss = a + sm[sel]
                    xi = 2 * (ss - a) / (b - a) - 1
                    out[sel] += k * C.chebval(xi, cf) * np.exp(1j * k * phases(ag, d, ss, kind, info)[0])
        return out


def gl_grid(charts, k, nmin=32):
    S, W = [], []
    for (a, b, _) in charts:
        nq = int(max(nmin, 2.2 * k * (b - a) / np.pi * 2 + 24))
        s, w, _ = gl_nodes(a, b, nq); S.append(s); W.append(w)
    return np.concatenate(S), np.concatenate(W)


def evaluate(sur, cv, ag, vnodes, k, alpha, xh, oracle_coef=None):
    """Errors of base-only, surrogate and (optionally) oracle densities on GL nodes."""
    d = np.array([np.cos(alpha), np.sin(alpha)])
    charts, info, cf = sur.coefs(cv, ag, k, alpha)
    s, w = gl_grid(charts, k)
    t, x, _, _ = ag.point(s)
    vex = trig_eval(vnodes, t)
    E = np.exp(-1j * k * (np.outer(xh[0], x[0]) + np.outer(xh[1], x[1]))) * w
    ffex = E @ vex
    res = {}
    bk = sur.base
    cands = {"base": Surrogate.density(ag, s, k, d, charts, info, cf, corr=False, base_kind=bk),
             "surrogate": Surrogate.density(ag, s, k, d, charts, info, cf, base_kind=bk)}
    if oracle_coef is not None and bk == "fock":
        cands["oracle"] = Surrogate.density(ag, s, k, d, charts, info, oracle_coef)
    nv = np.sqrt((w * abs(vex) ** 2).sum())
    for name, vh in cands.items():
        res[f"{name}_dens"] = float(np.sqrt((w * abs(vh - vex) ** 2).sum()) / nv)
        ff = E @ vh
        res[f"{name}_ff_l2"] = float(np.linalg.norm(ff - ffex) / np.linalg.norm(ffex))
        res[f"{name}_ff_max"] = float(abs(ff - ffex).max() / abs(ffex).max())
    return res
