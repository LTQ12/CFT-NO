"""Galerkin hybrid numerical-asymptotic BEMs for 2D sound-soft smooth convex scattering, reproduced from
  [EE18] F. Ecevit, H. H. Eruslu, "A Galerkin BEM for high-frequency scattering problems based on
         frequency-dependent changes of variables", IMA J. Numer. Anal. (2018) 893-923, arXiv:1609.02216.
  [EO17] F. Ecevit, H. C. Ozen, "Frequency-adapted Galerkin boundary element methods for convex scattering
         problems", Numer. Math. (2017); space of Sec. 3 / Eq. (8) of arXiv:1609.02216.

What follows the papers:
  * unknown eta = du/dn, ansatz eta = exp(ik alpha.x) eta_slow, CFIE R_k = 1/2 I + D'_k - ik S_k,
    f = du_inc/dn - ik u_inc (the CFIE of the numerical section of arXiv:1609.02216);
  * arclength parametrisation gamma with alpha.nu(gamma(0)) = 1, shadow boundaries t1 < t2;
  * EE18 space A^C_d: intervals I_IT1, I_IT2, I_ST1, I_ST2 (transition, with the change of variables
    phi(s) = t1 + varphi(s) k^psi(s) etc., psi from -1/3 to 0), I_SB1, I_SB2, and (J = 8) I_IL, I_DS;
    polynomials P_d o phi^{-1} on transition intervals and P_d elsewhere;
  * EO17 space A_d: 4m intervals with endpoints t1 + xi k^{-1/3+eps_j}, eps_j = (1/3)(2m-2j+1)/(2m+1);
  * Galerkin integrals: Nystrom (inner) and trapezoidal (outer) rules at about 10-12 points per wavelength,
    with a smooth partition of unity on the region boundaries;
  * parameter set-up of the circle experiments: xi' and zeta' reach the specular point (alpha.gamma = -1
    side) and the shadow centre (alpha.nu = 1), xi = xi'/2, zeta = zeta'/2, then tuned (see tune()).
Not specified in the papers, chosen here (documented):
  * the partition of unity: C-infinity steps of half-width BETA * (shorter neighbouring interval), BETA = 0.25;
  * polynomial basis: Legendre polynomials of the (mapped) local variable on the support of each POU function;
  * the tuning: a coarse search of the two ratios xi/xi', zeta/zeta' on a separate tuning shape (not a test shape).
"""
import os, sys, time, numpy as np
from numpy.polynomial import legendre as Lg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bie import assemble

BETA = 0.25
PPW = 11.0


def smooth_step(u):
    """C-infinity step: 0 for u <= 0, 1 for u >= 1."""
    u = np.clip(u, 0, 1)
    a = np.where(u > 0, np.exp(-1 / np.maximum(u, 1e-300)), 0.0)
    b = np.where(u < 1, np.exp(-1 / np.maximum(1 - u, 1e-300)), 0.0)
    return a / (a + b)


class Boundary:
    """Nystrom grid (trapezoidal in the curve parameter) with arclength, alpha.nu, alpha.x."""

    def __init__(self, cv, k, alpha, ppw=PPW):
        from charts import ArcGeom
        ag = cv._ag if hasattr(cv, "_ag") else ArcGeom(cv)   # spectral arclength, cached on the curve
        cv._ag = ag; L = ag.L
        N = int(np.ceil(ppw * k * L / (2 * np.pi) / 2)) * 2
        self.geo = geo = cv.nodes(N); self.N, self.k, self.L = N, k, L
        self.w = geo["sp"] * 2 * np.pi / N
        s = ag.s_of_t(geo["t"])
        self.al = np.array([np.cos(alpha), np.sin(alpha)])
        self.dn = self.al @ geo["n"]; self.dx = self.al @ geo["x"]
        # gamma(0): the point with alpha.nu = 1 (shadow centre); arclength measured from there
        tf = 2 * np.pi * np.arange(1 << 15) / (1 << 15); fine = cv.nodes(1 << 15); dnf = self.al @ fine["n"]
        sf = ag.s_of_t(tf)
        i0 = int(np.argmax(dnf)); s0 = sf[i0]
        self.s = (s - s0) % L
        sfr = (sf - s0) % L; o = np.argsort(sfr); sfr, dno = sfr[o], dnf[o]
        z = np.nonzero(np.diff(np.sign(dno)))[0]
        roots = [sfr[i] - dno[i] * (sfr[i + 1] - sfr[i]) / (dno[i + 1] - dno[i]) for i in z]
        assert len(roots) == 2, roots
        self.t1, self.t2 = sorted(roots)
        self.s_spec = sfr[int(np.argmin(dno))]          # specular point (alpha.nu = -1)
        self.sf, self.dnf = sfr, dno


def ee18_intervals(b, k, xi1, xi1p, zeta1, zeta1p, xi2, xi2p, zeta2, zeta2p, J):
    """EE18 intervals: list of (a, b, kind, info); kind in {'IT1','IT2','ST1','ST2','SB1','SB2','IL','DS'}."""
    t1, t2, L = b.t1, b.t2, b.L; c = k ** (-1 / 3)
    I = [(t1 - zeta1 * c, t1 + xi1 * c, "SB1"), (t1 + xi1 * c, t1 + xi1p, "IT1")]
    if J == 8:
        I.append((t1 + xi1p, t2 - xi2p, "IL"))
    I += [(t2 - xi2p, t2 - xi2 * c, "IT2"), (t2 - xi2 * c, t2 + zeta2 * c, "SB2"), (t2 + zeta2 * c, t2 + zeta2p, "ST2")]
    if J == 8:
        I.append((t2 + zeta2p, L + t1 - zeta1p, "DS"))
    I.append((L + t1 - zeta1p, L + t1 - zeta1 * c, "ST1"))
    return I


def ee18_phi(kind, s, a, bb, k, b, P):
    """The change of variables phi on a transition interval; outside [a, bb] (overlap of the partition of
    unity) it is continued linearly with the end slope, which keeps it monotone."""
    s = np.asarray(s, float)
    inside = _ee18_phi(kind, np.clip(s, a, bb), a, bb, k, b, P)
    h = 1e-7 * (bb - a)
    sa = (_ee18_phi(kind, np.array([a + h]), a, bb, k, b, P) - _ee18_phi(kind, np.array([a]), a, bb, k, b, P)) / h
    sb = (_ee18_phi(kind, np.array([bb]), a, bb, k, b, P) - _ee18_phi(kind, np.array([bb - h]), a, bb, k, b, P)) / h
    return inside + np.where(s < a, (s - a) * sa, 0.0) + np.where(s > bb, (s - bb) * sb, 0.0)


def _ee18_phi(kind, s, a, bb, k, b, P):
    xi1, xi1p, zeta1, zeta1p, xi2, xi2p, zeta2, zeta2p = P
    u = (s - a) / (bb - a)
    t1, t2, L = b.t1, b.t2, b.L
    if kind == "IT1":
        vp = xi1 + (xi1p - xi1) * u; ps = -(1 - u) / 3; return t1 + vp * k ** ps
    if kind == "IT2":
        vp = xi2p + (xi2 - xi2p) * u; ps = -u / 3; return t2 - vp * k ** ps
    if kind == "ST1":
        vp = zeta1p + (zeta1 - zeta1p) * u; ps = -u / 3; return L + t1 - vp * k ** ps
    vp = zeta2 + (zeta2p - zeta2) * u; ps = -(1 - u) / 3; return t2 + vp * k ** ps     # ST2


def eo17_intervals(b, k, m, xi1, zeta1, xi2, zeta2):
    t1, t2, L = b.t1, b.t2, b.L
    eps = [(2 * m - 2 * j + 1) / (3 * (2 * m + 1)) for j in range(1, m + 1)]
    e = lambda j: k ** (-1 / 3 + eps[j - 1])
    I = [(t1 - zeta1 * e(m), t1 + xi1 * e(m), "SB1")]
    for j in range(m - 1, 0, -1):
        I.append((t1 + xi1 * e(j + 1), t1 + xi1 * e(j), "IT1"))
    I.append((t1 + xi1 * e(1), t2 - xi2 * e(1), "IL"))
    for j in range(1, m):
        I.append((t2 - xi2 * e(j), t2 - xi2 * e(j + 1), "IT2"))
    I.append((t2 - xi2 * e(m), t2 + zeta2 * e(m), "SB2"))
    for j in range(m - 1, 0, -1):
        I.append((t2 + zeta2 * e(j + 1), t2 + zeta2 * e(j), "ST2"))
    I.append((t2 + zeta2 * e(1), L + t1 - zeta1 * e(1), "DS"))
    for j in range(1, m):
        I.append((L + t1 - zeta1 * e(j), L + t1 - zeta1 * e(j + 1), "ST1"))
    return I


def basis(b, k, intervals, d, cov=None):
    """Basis functions at the Nystrom nodes: (N, DOF). cov(kind, s, a, bb) -> phi(s) for change-of-variables
    intervals (EE18) or None (EO17 / plain polynomials)."""
    L = b.L; s0 = intervals[0][0]
    ss = (b.s - s0) % L + s0                          # node arclengths unwrapped onto [s0, s0 + L)
    ends = [iv[0] for iv in intervals] + [intervals[0][0] + L]
    lens = np.diff(ends); n = len(intervals)
    for (a, bb, _), ln in zip(intervals, lens):
        assert bb > a and abs((bb - a) - ln) < 1e-9, (a, bb, ln)
    dl = [BETA * min(lens[i - 1], lens[i]) for i in range(n)]      # overlap half-width at the left end of interval i
    cols = []; ph = np.exp(1j * k * b.dx)
    for i, (a, bb, kind) in enumerate(intervals):
        lo, hi = a - dl[i], bb + dl[(i + 1) % n]
        x = (ss - lo) % L + lo                          # node positions relative to the support start
        on = (x >= lo) & (x <= hi)
        up = smooth_step((x - (a - dl[i])) / (2 * dl[i]))
        down = 1 - smooth_step((x - (bb - dl[(i + 1) % n])) / (2 * dl[(i + 1) % n]))
        chi = np.where(on, up * down, 0.0)
        if cov is not None and kind in ("IT1", "IT2", "ST1", "ST2"):
            g = np.linspace(lo, hi, 4001); fg = cov(kind, g, a, bb)
            if fg[-1] < fg[0]:
                g, fg = g[::-1], fg[::-1]
            assert np.all(np.diff(fg) > 0), "change of variables not monotone on the extended support"
            # local variable sigma = phi^{-1}(s), mapped affinely to [-1, 1] over the support
            sig = np.interp(x, fg, g)
            sl, sh = np.interp([lo, hi], fg, g)
            sig_lo, sig_hi = min(sl, sh), max(sl, sh)
            loc = 2 * (sig - sig_lo) / (sig_hi - sig_lo) - 1
        else:
            loc = 2 * (x - lo) / (hi - lo) - 1
        loc = np.clip(loc, -1, 1)
        V = Lg.legvander(loc, d)
        cols.append(chi[:, None] * ph[:, None] * V)
    return np.concatenate(cols, 1)


def galerkin_solve(b, A, f, B):
    """Galerkin projection with the trapezoidal outer rule: (B^H W A B) c = B^H W f."""
    BW = B.conj().T * b.w[None, :]
    G = BW @ (A @ B); r = BW @ f
    c = np.linalg.lstsq(G, r, rcond=None)[0]           # lstsq: the Legendre bases on overlapping supports
    return B @ c, np.linalg.cond(G)


def cfie(b, k, mats=None):
    mats = mats or assemble(b.geo, k, need=("S", "Dp"))
    A = 0.5 * np.eye(b.N) + mats["Dp"] - 1j * k * mats["S"]
    x, n = b.geo["x"], b.geo["n"]
    ui = np.exp(1j * k * (b.al @ x))
    f = 1j * k * (b.al @ n) * ui - 1j * k * ui
    return A, f


def default_params(b, r_xi=0.5, r_zeta=0.5):
    """Circle set-up of EE18 Sec. 5 for a general convex curve: xi' to the specular point, zeta' to the shadow
    centre (alpha.nu = 1, s = 0), xi = r_xi xi', zeta = r_zeta zeta'."""
    xi1p = b.s_spec - b.t1; xi2p = b.t2 - b.s_spec
    zeta1p = b.t1; zeta2p = b.L - b.t2
    return (r_xi * xi1p, xi1p, r_zeta * zeta1p, zeta1p, r_xi * xi2p, xi2p, r_zeta * zeta2p, zeta2p)


def solve_ee18(b, k, d, P, A, f, J=6):
    I = ee18_intervals(b, k, *P, J)
    cov = lambda kind, s, a, bb: ee18_phi(kind, s, a, bb, k, b, P)
    B = basis(b, k, I, d, cov)
    v, cnd = galerkin_solve(b, A, f, B)
    return v, B.shape[1], cnd


def solve_eo17(b, k, d, m, A, f, c_xi=1.0, c_zeta=1.0):
    """EO17 space with k-independent lengths xi = c_xi * (lit half-arc), zeta = c_zeta * (shadow half-arc):
    the outermost transition points are t1 + xi k^{-1/3+eps_1} etc. (eps_1 = (2m-1)/(3(2m+1)))."""
    xi1, xi2 = c_xi * (b.s_spec - b.t1), c_xi * (b.t2 - b.s_spec)
    z1, z2 = c_zeta * b.t1, c_zeta * (b.L - b.t2)
    I = eo17_intervals(b, k, m, xi1, z1, xi2, z2)
    B = basis(b, k, I, d, None)
    v, cnd = galerkin_solve(b, A, f, B)
    return v, B.shape[1], cnd
