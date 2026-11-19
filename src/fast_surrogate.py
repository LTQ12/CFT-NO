"""Vectorised surrogate evaluation (same model and same features as surrogate.py, without Python
loops over charts). Per shape: tables and quadrature grid; per incidence: O(N_q) array work."""
import os
import sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from charts import ArcGeom, graded_edges
from g2_data import FB, M, N_LIT, N_SH, radial_descriptor
from numpy.polynomial import chebyshev as C

FF_CONST = lambda k: -np.exp(1j * np.pi / 4) / np.sqrt(8 * np.pi * k)


class ShapeEval:
    def __init__(self, cv, k, ppw=5.0, Nf=2 ** 13):
        self.cv, self.k = cv, k
        ag = ArcGeom(cv, Nf)
        d1, d2 = cv.dz(ag.t_f), cv.ddz(ag.t_f)
        sp = np.hypot(d1[0], d1[1])
        self.kap_f = (d1[0] * d2[1] - d1[1] * d2[0]) / sp ** 3
        g = np.abs(self.kap_f) ** (2 / 3)
        ag.I_f = np.concatenate([[0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(ag.s_f))])
        self.ag, self.L, self.Itot = ag, ag.L, ag.I_f[-1]
        g0 = cv.nodes(512)
        Nq = int(ppw * k * g0["sp"].max()) // 2 * 2
        tq = 2 * np.pi * np.arange(Nq) / Nq
        e1, e2 = cv.dz(tq), cv.ddz(tq)
        spq = np.hypot(e1[0], e1[1])
        self.xq = cv.z(tq); self.wq = spq * 2 * np.pi / Nq
        # arclength at the nodes by exact integration of the trigonometric interpolant of |z'|
        cf = np.fft.fft(spq) / Nq; m = np.fft.fftfreq(Nq, 1.0 / Nq); cf[Nq // 2] = 0
        mm = np.where(m == 0, 1, m)
        g = np.where(m == 0, 0, cf / (1j * mm))
        self.sq = cf[0].real * tq + Nq * np.fft.ifft(g) - g.sum()
        self.sq = np.real(self.sq)
        ag.L = float(cf[0].real * 2 * np.pi); self.psiq = np.arctan2(e1[1], e1[0])
        self.kapq = (e1[0] * e2[1] - e1[1] * e2[0]) / spq ** 3
        self.Iq = np.interp(self.sq, ag.s_f, ag.I_f)
        self.Nq = Nq

    def _I(self, s):
        return np.interp(np.mod(s, self.L), self.ag.s_f, self.ag.I_f)

    def _kap(self, s):
        return np.interp(np.mod(s, self.L), self.ag.s_f, self.kap_f)

    def _psi(self, s):
        return np.interp(np.mod(s, self.L), self.ag.s_f, self.ag.psi_f)

    def charts(self, alpha):
        ag, L, k = self.ag, self.L, self.k
        p0 = ag.psi_f[0]
        s_in = np.interp(p0 + (alpha - p0) % (2 * np.pi), ag.psi_f, ag.s_f)
        s_out = np.interp(p0 + (alpha + np.pi - p0) % (2 * np.pi), ag.psi_f, ag.s_f)
        Ls = (s_out - s_in) % L; Ll = L - Ls
        kap = self._kap(np.array([s_in, s_out])); sF = (2 / kap ** 2 / k) ** (1 / 3)
        e1 = graded_edges(Ls / 2, N_SH, sF[0]); e2 = graded_edges(Ls / 2, N_SH, sF[1])
        e3 = graded_edges(Ll / 2, N_LIT, sF[1]); e4 = graded_edges(Ll / 2, N_LIT, sF[0])
        a = np.concatenate([s_in + e1[:-1], s_out - e2[1:], s_out + e3[:-1], s_in - e4[1:]])
        b = np.concatenate([s_in + e1[1:], s_out - e2[:-1], s_out + e3[1:], s_in - e4[:-1]])
        kind = np.repeat([0, 1, 2, 3], [N_SH, N_SH, N_LIT, N_LIT])   # sh_in, sh_out, lit_out, lit_in
        return dict(a=a, b=b, kind=kind, s_in=s_in, s_out=s_out, alpha=alpha)

    def features(self, ch, desc=None):
        k, a, b, kind = self.k, ch["a"], ch["b"], ch["kind"]
        side = np.where((kind == 0) | (kind == 3), 1.0, -1.0)
        sb = np.where(side > 0, ch["s_in"], ch["s_out"])
        shadow = kind <= 1
        c = (k / 2) ** (1 / 3); It = self.Itot
        def xi_of(q):
            fwd = (self._I(q) - self._I(sb)) % It; bwd = (self._I(sb) - self._I(q)) % It
            return np.where(shadow, c * np.where(side > 0, fwd, bwd), -c * np.where(side > 0, bwd, fwd))
        xa, xb = xi_of(a + 1e-12), xi_of(b - 1e-12)
        jj = np.cos(np.pi * (np.arange(5) + 0.5) / 5)
        q = 0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * jj[None, :]
        al = ch["alpha"]
        dn = np.sin(self._psi(np.stack([a, 0.5 * (a + b), b], 1)) - al)
        X = np.concatenate([np.stack([shadow.astype(float), side, np.arcsinh(xa / 2), np.arcsinh(xb / 2),
                                      np.log((b - a) * k ** (1 / 3)), np.full(32, np.log(min(k, float(os.environ.get("K_FEAT_MAX", "inf"))) / 64)),
                                      np.log(self._kap(sb))], 1), np.log(self._kap(q)), dn], 1)
        if desc is not None:
            X = np.concatenate([X, np.tile(desc, (32, 1))], 1)
        return X

    def density(self, ch, coef, base=True, base_kind="fock"):
        k, L = self.k, self.L
        al = ch["alpha"]; d = np.array([np.cos(al), np.sin(al)])
        s = self.sq; dx = d @ self.xq; dn = np.sin(self.psiq - al)
        s_in, s_out = ch["s_in"], ch["s_out"]
        ti = self.ag.t_of_s(np.array([s_in, s_out])); xb = self.cv.z(ti); dxb = d @ xb
        out = np.zeros(len(s), complex)
        if base:
            out += np.where(dn < 0, 2j * k * dn * np.exp(1j * k * dx), 0)
        if base and base_kind == "fock":
            scale = k * (2 * self.kapq / k) ** (1 / 3); c = (k / 2) ** (1 / 3); It = self.Itot
            Ls = (s_out - s_in) % L; insh = ((s - s_in) % L) < Ls
            Iin, Iout, Is = self._I(s_in), self._I(s_out), self.Iq
            for xs, xl, phc in [(c * ((Is - Iin) % It), -c * ((Iin - Is) % It), dxb[0] + (s - s_in) % L),
                                (c * ((Iout - Is) % It), -c * ((Is - Iout) % It), dxb[1] + (s_out - s) % L)]:
                term = np.where(insh, FB._interp(FB.G, np.where(insh, xs, 0)) * np.exp(1j * k * phc),
                                FB._interp(FB.F, np.where(insh, -1.0, xl)) * np.exp(1j * k * dx))
                out += scale * term
        # chart corrections
        a, b, kind = ch["a"], ch["b"], ch["kind"]
        ua = (a - s_in) % L
        order = np.argsort(ua)
        u = (s - s_in) % L
        idx = order[np.clip(np.searchsorted(ua[order], u, side="right") - 1, 0, 31)]
        xi = np.clip(2 * (u - ua[idx]) / (b - a)[idx] - 1, -1, 1)
        T = C.chebvander(xi, M - 1)
        val = (T * coef[idx]).sum(1)
        kk = kind[idx]
        ph = np.where(kk >= 2, dx, np.where(kk == 0, dxb[0] + (s - s_in) % L, dxb[1] + (s_out - s) % L))
        return out + k * val * np.exp(1j * k * ph)

    def farfield_operator(self, xh):
        return FF_CONST(self.k) * np.exp(-1j * self.k * (np.outer(xh[0], self.xq[0]) + np.outer(xh[1], self.xq[1]))) * self.wq


def surrogate_farfield(sur, cv, k, alphas, xh, se=None, E=None, timing=None):
    import time
    T = timing if timing is not None else {}
    t0 = time.perf_counter()
    se = se or ShapeEval(cv, k)
    E = E if E is not None else se.farfield_operator(xh)
    T["setup"] = T.get("setup", 0) + time.perf_counter() - t0
    t0 = time.perf_counter()
    chs = [se.charts(al) for al in alphas]
    desc = None
    if "local" not in sur.flags:
        desc = [radial_descriptor(se.ag, al) for al in alphas]
    X = np.stack([se.features(ch, None if desc is None else desc[i]) for i, ch in enumerate(chs)])
    T["features"] = T.get("features", 0) + time.perf_counter() - t0
    t0 = time.perf_counter()
    Xraw = X
    if "dimless" in sur.flags:
        import dimless; dimless.DOD = "dod" in sur.flags
        from dimless import to_dimless
        X = to_dimless(X)
    if "ctx" in sur.flags:
        from g2_data import add_context
        Xp = add_context(X)          # X already has no descriptor when the model is "local"
        if "dimless" in sur.flags:
            from dimless import gate_context
            Xp = gate_context(Xp, Xraw)
    else:
        Xp = X
    Yn = sur.net.forward(((Xp.reshape(-1, Xp.shape[-1]) - sur.xm) / sur.xs).astype(np.float32)).astype(np.float64)
    Yn = Yn.reshape(len(alphas), 32, -1)
    T["network"] = T.get("network", 0) + time.perf_counter() - t0
    t0 = time.perf_counter()
    V = np.empty((se.Nq, len(alphas)), complex)
    for j, ch in enumerate(chs):
        W = ch["b"] - ch["a"]
        if "dimless" in sur.flags:
            from dimless import scale_factor
            Y = Yn[j] * sur.ys / scale_factor(k, W, Xraw[j], sur.p)[:, None]
        else:
            Y = Yn[j] * sur.ys / (k ** sur.p * np.sqrt(W)[:, None])
        if os.environ.get("FOCK_ONLY"):                  # baseline: Fock current alone (network switched off)
            Y = np.zeros_like(Y)
        V[:, j] = se.density(ch, Y[:, :M] + 1j * Y[:, M:], base_kind=getattr(sur, "base", "fock"))
    T["density"] = T.get("density", 0) + time.perf_counter() - t0
    t0 = time.perf_counter(); F = E @ V; T["farfield"] = T.get("farfield", 0) + time.perf_counter() - t0
    return F, V, se
