"""Phase-separated chart representation of the Neumann trace on a convex curve.

v(s)/k = A0(s) exp(ik d.x(s))                            on lit charts
v(s)/k = A+(s) exp(ik phi+(s)) + A-(s) exp(ik phi-(s))   on shadow charts
phi+- = d.x(s_sb+-) + |s - s_sb+-|  (creeping-wave phase measured along the boundary)
Each amplitude is a Chebyshev series of degree M-1 on its chart.
Charts are graded geometrically away from the two shadow-boundary points, with first
width c0 * s_F, s_F = (2 rho^2 / k)^(1/3) the Fock length at that point.
"""
import numpy as np
from numpy.polynomial import chebyshev as C


class ArcGeom:
    def __init__(self, cv, Nf=2 ** 16):
        # arclength by exact integration of the trigonometric interpolant of |z'| (spectral);
        # inverse map t(s) by cubic spline (errors ~1e-12 instead of O(h^2))
        from scipy.interpolate import CubicSpline
        t = 2 * np.pi * np.arange(Nf + 1) / Nf
        d1 = cv.dz(t)
        Nc = 4096
        tc = 2 * np.pi * np.arange(Nc) / Nc
        e = cv.dz(tc); spc = np.hypot(e[0], e[1])
        cf = np.fft.rfft(spc) / Nc
        c0 = cf[0].real
        j = np.arange(1, len(cf)); keep = np.abs(cf[1:]) > 1e-16 * abs(cf[0])
        j, g = j[keep], cf[1:][keep]
        s = c0 * t
        for i in range(0, len(t), 4096):
            tt = t[i:i + 4096]
            s[i:i + 4096] += 2 * ((np.exp(1j * np.outer(tt, j)) - 1) @ (g / (1j * j))).real
        self.cv, self.t_f, self.s_f, self.L = cv, t, s, 2 * np.pi * c0
        self.psi_f = np.unwrap(np.arctan2(d1[1], d1[0]))      # tangent angle, increasing (convex, ccw)
        self._t_of_s = CubicSpline(s, t)
        self._s_of_t = CubicSpline(t, s)

    def t_of_s(self, s):
        return self._t_of_s(np.mod(s, self.L))

    def s_of_t(self, t):
        return self._s_of_t(np.mod(t, 2 * np.pi))

    def point(self, s):
        t = self.t_of_s(s)
        x, d1, d2 = self.cv.z(t), self.cv.dz(t), self.cv.ddz(t)
        sp = np.hypot(d1[0], d1[1])
        tau = d1 / sp
        n = np.array([tau[1], -tau[0]])
        kap = (d1[0] * d2[1] - d1[1] * d2[0]) / sp ** 3
        return t, x, n, kap


def shadow_points(ag, d):
    """Fast version for convex curves: s_in where the tangent angle equals arg(d) (mod 2pi),
    s_out where it equals arg(d) + pi."""
    al = np.arctan2(d[1], d[0])
    p0, p1 = ag.psi_f[0], ag.psi_f[-1]
    if p1 - p0 > 2 * np.pi - 1e-6 and np.all(np.diff(ag.psi_f) > 0):
        def s_at(target):
            tgt = p0 + (target - p0) % (2 * np.pi)
            return np.interp(tgt, ag.psi_f, ag.s_f)
        return s_at(al), s_at(al + np.pi)
    return shadow_points_bisect(ag, d)


def shadow_points_bisect(ag, d):
    """Arclengths of the two points where d.n = 0; returns (s_enter, s_exit) such that
    the shadow arc runs from s_enter to s_exit in increasing s (mod L)."""
    s = np.linspace(0, ag.L, 20001)[:-1]
    _, x, n, _ = ag.point(s)
    g = d[0] * n[0] + d[1] * n[1]          # >0 shadow (outward normal along d), <0 lit
    roots = []
    for i in np.nonzero(np.sign(g) != np.sign(np.roll(g, -1)))[0]:
        a, b = s[i], s[i] + (s[1] - s[0])
        for _ in range(60):
            m = 0.5 * (a + b)
            _, _, nm, _ = ag.point(np.array([m]))
            _, _, na, _ = ag.point(np.array([a]))
            if np.sign(d @ nm[:, 0]) == np.sign(d @ na[:, 0]):
                a = m
            else:
                b = m
        roots.append((0.5 * (a + b), g[i] < 0))   # g goes - -> + : entering shadow
    assert len(roots) == 2, roots
    s_enter = [r for r, e in roots if e][0]
    s_exit = [r for r, e in roots if not e][0]
    return s_enter, s_exit


SMOOTH_GRADING = False      # True: one smooth grading law (no switch to uniform spacing)


def graded_edges(length, n, w0):
    """n intervals covering [0, length], first width w0, geometric growth with ratio q.
    Default (legacy): uniform spacing when n*w0 >= length.  With SMOOTH_GRADING the ratio q > 0 solves
    w0 (q^n - 1)/(q - 1) = length for every length, so the edges depend smoothly on (length, w0)."""
    if not SMOOTH_GRADING:
        if n * w0 >= length:
            return np.linspace(0, length, n + 1)
        lo, hi = 1.0, 10.0
        f = lambda q: w0 * (q ** n - 1) / (q - 1) - length
        while f(hi) < 0:
            hi *= 2
        for _ in range(100):
            m = 0.5 * (lo + hi)
            lo, hi = (m, hi) if f(m) < 0 else (lo, m)
        q = 0.5 * (lo + hi)
    else:
        # g(u) = sum_{i<n} e^{u i} is increasing in u = log q; solve w0 g(u) = length by Newton (smooth in data)
        target = length / w0
        u = 0.0 if abs(target - n) < 1e-14 else np.log(max(target, 1e-3) / n) * 2 / max(n - 1, 1)
        i = np.arange(n)
        for _ in range(60):
            e = np.exp(u * i); g = e.sum(); dg = (i * e).sum()
            step = (g - target) / dg
            u -= step
            if abs(step) < 1e-15:
                break
        q = np.exp(u)
    e = np.concatenate([[0], np.cumsum(w0 * q ** np.arange(n))])
    return e * length / e[-1]


def build_charts(ag, d, k, n_lit=9, n_sh=7, c0=1.0, mode="full"):
    """Return list of charts (s_a, s_b, kind). kind in {'lit','shadow'}.
    mode='full': graded, shadow charts two-phase; 'graded_inc': graded, all incident phase;
    'uniform_inc': 2*(n_lit+n_sh) equal-arclength charts, incident phase."""
    L = ag.L
    if mode == "uniform_inc":
        nC = 2 * (n_lit + n_sh)
        e = np.linspace(0, L, nC + 1)
        return [(e[i], e[i + 1], "lit") for i in range(nC)], None
    s_in, s_out = shadow_points(ag, d)
    Ls = (s_out - s_in) % L
    Ll = L - Ls
    _, _, _, kap = ag.point(np.array([s_in, s_out]))
    sF = (2 / kap ** 2 / k) ** (1 / 3)
    charts = []
    # shadow arc: half from s_in (forward), half from s_out (backward)
    kin, kout = {"full": ("shadow", "shadow"), "full1": ("sh_in", "sh_out")}.get(mode, ("lit", "lit"))
    e = graded_edges(Ls / 2, n_sh, c0 * sF[0])
    charts += [(s_in + e[i], s_in + e[i + 1], kin) for i in range(n_sh)]
    e = graded_edges(Ls / 2, n_sh, c0 * sF[1])
    charts += [(s_out - e[i + 1], s_out - e[i], kout) for i in range(n_sh)]
    # lit arc: from s_out forward to s_out + Ll/2, and from s_in backward
    e = graded_edges(Ll / 2, n_lit, c0 * sF[1])
    charts += [(s_out + e[i], s_out + e[i + 1], "lit") for i in range(n_lit)]
    e = graded_edges(Ll / 2, n_lit, c0 * sF[0])
    charts += [(s_in - e[i + 1], s_in - e[i], "lit") for i in range(n_lit)]
    info = dict(s_in=s_in, s_out=s_out, Ls=Ls, sF=sF.tolist(), kap_sb=kap.tolist())
    return charts, info


def phases(ag, d, s, kind, info):
    """List of phase functions (arrays) for the chart kind at arclengths s (unwrapped)."""
    _, x, _, _ = ag.point(s)
    if kind == "lit":
        return [d[0] * x[0] + d[1] * x[1]]
    _, xb, _, _ = ag.point(np.array([info["s_in"], info["s_out"]]))
    L = ag.L
    dp = (s - info["s_in"]) % L                     # distance travelled from entry point
    dm = (info["s_out"] - s) % L                    # distance from exit point (going backward)
    if kind == "sh_in":
        return [d @ xb[:, 0] + dp]
    if kind == "sh_out":
        return [d @ xb[:, 1] + dm]
    return [d @ xb[:, 0] + dp, d @ xb[:, 1] + dm]


def trig_eval(vnodes, t, chunk=2000):
    """Evaluate trigonometric interpolant of nodal values (N equispaced) at points t."""
    N = len(vnodes)
    c = np.fft.fft(vnodes, axis=0) / N
    m = np.fft.fftfreq(N, 1.0 / N)
    c[N // 2] = 0
    out = np.empty((len(t),) + vnodes.shape[1:], complex)
    for i in range(0, len(t), chunk):
        out[i:i + chunk] = np.exp(1j * np.outer(t[i:i + chunk], m)) @ c
    return out


def gl_nodes(a, b, n):
    xg, wg = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * xg + 0.5 * (a + b), 0.5 * (b - a) * wg, xg


def fit_and_eval(ag, vnodes, k, d, charts, info, M, xhat=None):
    """Least-squares fit of each chart; returns reconstruction on GL nodes, exact values, weights,
    positions (for far field) and the coefficient count."""
    rec, ex, W, X, ncoef = [], [], [], [], 0
    for (a, b, kind) in charts:
        width = b - a
        nq = int(max(4 * M, 2.2 * k * width / np.pi * 2 + 3 * M))
        s, w, xi = gl_nodes(a, b, nq)
        t, x, _, _ = ag.point(s)
        vex = trig_eval(vnodes, t) / k
        T = C.chebvander(xi, M - 1)
        ph = phases(ag, d, s, kind, info)
        B = np.concatenate([T * np.exp(1j * k * p)[:, None] for p in ph], axis=1)
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(B * sw[:, None], vex * sw, rcond=None)[0]
        rec.append(B @ coef); ex.append(vex); W.append(w); X.append(x)
        ncoef += B.shape[1]
    rec, ex, W = np.concatenate(rec), np.concatenate(ex), np.concatenate(W)
    X = np.concatenate(X, axis=1)
    err = np.sqrt((W * abs(rec - ex) ** 2).sum() / (W * abs(ex) ** 2).sum())
    out = dict(dens_rel_L2=float(err), ncoef=int(ncoef))
    if xhat is not None:
        E = np.exp(-1j * k * (np.outer(xhat[0], X[0]) + np.outer(xhat[1], X[1])))
        c = -np.exp(1j * np.pi / 4) / np.sqrt(8 * np.pi * k) * k
        f_rec = c * (E * W[None, :]) @ rec
        f_ex = c * (E * W[None, :]) @ ex
        out["ff_rel_L2"] = float(np.linalg.norm(f_rec - f_ex) / np.linalg.norm(f_ex))
        out["ff_rel_max"] = float(abs(f_rec - f_ex).max() / abs(f_ex).max())
        out["_ff_ex"] = f_ex
    return out
