"""Nystrom solver (Kress log-splitting quadrature) for 2D sound-soft scattering.

Unknown: Neumann trace v = d_n u of the total field on Gamma (direct formulation).
Operators on a smooth closed curve z(t), t in [0, 2pi), 2n equispaced nodes:
  S v(x)  = int Phi(x,y) v(y) ds_y               single layer
  Dp v(x) = int d_{n_x} Phi(x,y) v(y) ds_y        adjoint double layer
  T v(x)  = d_s (S v)(x)                         tangential derivative (spectral)
Star-combined operator (Spence, Chandler-Wilde, Graham, Smyshlyaev 2011):
  A v = (x.n)(v/2 + Dp v) + (x.t) T v - i eta S v,   eta = k|x| + i/2
  f   = x.grad u_inc - i eta u_inc
Re(A phi, phi) >= gamma ||phi||^2 with gamma = 0.5 * min(x.n) for star-shaped Gamma.
Standard direct CFIE (Burton-Miller type): (v/2 + Dp v) - i k S v = d_n u_inc - i k u_inc.
"""
import numpy as np
from scipy.special import hankel1, jv
j0 = lambda z: jv(0, z)          # jv accepts complex arguments (complex k for resonance checks)
j1 = lambda z: jv(1, z)

EULER = 0.57721566490153286


class Curve:
    """Closed curve given by callables z(t), z'(t), z''(t) returning arrays of shape (2, m)."""

    def __init__(self, z, dz, ddz, name=""):
        self.z, self.dz, self.ddz, self.name = z, dz, ddz, name

    def nodes(self, N):
        t = 2 * np.pi * np.arange(N) / N
        x, d1, d2 = self.z(t), self.dz(t), self.ddz(t)
        sp = np.hypot(d1[0], d1[1])                      # |z'|
        tau = d1 / sp                                    # unit tangent (ccw)
        nrm = np.array([tau[1], -tau[0]])                # outward normal
        kap = (d1[0] * d2[1] - d1[1] * d2[0]) / sp ** 3  # signed curvature
        return dict(t=t, x=x, sp=sp, tau=tau, n=nrm, kap=kap, N=N)


def star_curve(coefs_c, coefs_s, r0=1.0, name="star"):
    """r(t) = r0 + sum_m a_m cos(m t) + b_m sin(m t), z = r (cos t, sin t)."""
    a = np.asarray(coefs_c, float); b = np.asarray(coefs_s, float)
    m = np.arange(1, len(a) + 1)

    def r(t, d=0):
        c, s = np.cos(np.outer(t, m)), np.sin(np.outer(t, m))
        if d == 0:
            return r0 + c @ a + s @ b
        if d == 1:
            return -(s * m) @ a + (c * m) @ b
        return -(c * m ** 2) @ a - (s * m ** 2) @ b

    def z(t):
        return np.array([r(t) * np.cos(t), r(t) * np.sin(t)])

    def dz(t):
        R, R1 = r(t), r(t, 1)
        return np.array([R1 * np.cos(t) - R * np.sin(t), R1 * np.sin(t) + R * np.cos(t)])

    def ddz(t):
        R, R1, R2 = r(t), r(t, 1), r(t, 2)
        return np.array([R2 * np.cos(t) - 2 * R1 * np.sin(t) - R * np.cos(t),
                         R2 * np.sin(t) + 2 * R1 * np.cos(t) - R * np.sin(t)])

    cv = Curve(z, dz, ddz, name)
    cv.rfun = r
    cv.params = dict(r0=r0, a=a.tolist(), b=b.tolist())
    return cv


def ellipse(a, b, name="ellipse"):
    z = lambda t: np.array([a * np.cos(t), b * np.sin(t)])
    dz = lambda t: np.array([-a * np.sin(t), b * np.cos(t)])
    ddz = lambda t: np.array([-a * np.cos(t), -b * np.sin(t)])
    cv = Curve(z, dz, ddz, name)
    cv.params = dict(a=a, b=b)
    return cv


def kress_R(N):
    """Weights R_j for int ln(4 sin^2((t-tau)/2)) f(tau) dtau, as a circulant first row."""
    n = N // 2
    j = np.arange(N)
    tj = 2 * np.pi * j / N
    m = np.arange(1, n)
    Rrow = -(2 * np.pi / n) * (np.cos(np.outer(tj, m)) / m).sum(1) - (np.pi / n ** 2) * np.cos(n * tj)
    idx = (j[None, :] - j[:, None]) % N
    return Rrow[idx]


def spectral_diff(N):
    """Matrix of d/dt on N equispaced periodic nodes (Nyquist mode dropped)."""
    h = 2 * np.pi / N
    j = np.arange(N)
    col = np.zeros(N)
    jj = j[1:]
    col[1:] = 0.5 * (-1.0) ** jj / np.tan(jj * h / 2)
    idx = (j[:, None] - j[None, :]) % N
    return col[idx]


def _assemble_rows(geo, k, rows, need, Rfull):
    """Rows `rows` of the Nystrom matrices S and Dp (ds_y included)."""
    N = geo["N"]; x = geo["x"]; sp = geo["sp"]; nrm = geo["n"]; kap = geo["kap"]; t = geo["t"]
    dx = x[0][rows, None] - x[0][None, :]
    dy = x[1][rows, None] - x[1][None, :]
    r = np.hypot(dx, dy)
    diag = (np.arange(len(rows)), rows)
    r[diag] = 1.0
    kr = k * r
    E = np.zeros((len(rows), N)); E[diag] = 1.0
    L = np.log(4 * np.sin((t[rows, None] - t[None, :]) / 2) ** 2 + E)
    R = Rfull[rows]
    w = np.pi / (N // 2)
    out = {}
    if "S" in need:
        M1 = -(1 / (4 * np.pi)) * j0(kr) * sp[None, :]
        M = 0.25j * hankel1(0, kr) * sp[None, :]
        M2 = M - M1 * L
        del M
        M1[diag] = -(1 / (4 * np.pi)) * sp[rows]
        M2[diag] = (0.25j - EULER / (2 * np.pi) - np.log(k * sp[rows] / 2) / (2 * np.pi)) * sp[rows]
        out["S"] = R * M1 + w * M2
        del M1, M2
    if "Dp" in need:
        g = (dx * nrm[0][rows, None] + dy * nrm[1][rows, None]) / r * sp[None, :]
        K1 = (k / (4 * np.pi)) * j1(kr) * g
        K = -0.25j * k * hankel1(1, kr) * g
        K2 = K - K1 * L
        del K
        K1[diag] = 0.0
        K2[diag] = -kap[rows] * sp[rows] / (4 * np.pi)
        out["Dp"] = R * K1 + w * K2
    return out


def assemble(geo, k, need=("S", "Dp", "T"), block=512):
    """Nystrom matrices acting on nodal values of v (ds_y included), assembled in row blocks."""
    N = geo["N"]
    Rfull = kress_R(N)
    want = [m for m in ("S", "Dp") if m in need or (m == "S" and "T" in need)]
    out = {m: np.empty((N, N), complex) for m in want}
    for i0 in range(0, N, block):
        rows = np.arange(i0, min(N, i0 + block))
        blk = _assemble_rows(geo, k, rows, want, Rfull)
        for m in want:
            out[m][rows] = blk[m]
    del Rfull
    if "T" in need:
        out["T"] = (spectral_diff(N) @ out["S"]) / geo["sp"][:, None]
    return out


def star_combined(geo, k, mats=None):
    mats = mats or assemble(geo, k)
    N = geo["N"]; x = geo["x"]
    xn = (x * geo["n"]).sum(0); xt = (x * geo["tau"]).sum(0)
    eta = k * np.hypot(x[0], x[1]) + 0.5j
    A = xn[:, None] * (0.5 * np.eye(N) + mats["Dp"]) + xt[:, None] * mats["T"] - 1j * eta[:, None] * mats["S"]
    return A, eta, 0.5 * xn.min()


def rhs_star(geo, k, d, eta):
    x = geo["x"]
    ui = np.exp(1j * k * (d[0] * x[0] + d[1] * x[1]))
    xd = x[0] * d[0] + x[1] * d[1]
    return (1j * k * xd - 1j * eta) * ui


def solve_density(geo, k, dirs, mats=None, formulation="star"):
    """Return v (N x ndirs) for plane waves with directions dirs (2 x ndirs)."""
    mats = mats or assemble(geo, k, need=("S", "Dp") if formulation == "cfie" else ("S", "Dp", "T"))
    x = geo["x"]; nrm = geo["n"]
    ui = np.exp(1j * k * (dirs[0][None, :] * x[0][:, None] + dirs[1][None, :] * x[1][:, None]))
    if formulation == "star":
        A, eta, _ = star_combined(geo, k, mats)
        xd = x[0][:, None] * dirs[0][None, :] + x[1][:, None] * dirs[1][None, :]
        f = (1j * k * xd - 1j * eta[:, None]) * ui
    else:
        N = geo["N"]
        A = 0.5 * np.eye(N) + mats["Dp"] - 1j * k * mats["S"]
        dn = 1j * k * (nrm[0][:, None] * dirs[0][None, :] + nrm[1][:, None] * dirs[1][None, :])
        f = dn * ui - 1j * k * ui
    return np.linalg.solve(A, f)


def far_field(geo, k, v, xhat):
    """u_inf(xhat) = -e^{i pi/4}/sqrt(8 pi k) int e^{-ik xhat.y} v(y) ds (trapezoid, spectral)."""
    x = geo["x"]; w = geo["sp"] * 2 * np.pi / geo["N"]
    E = np.exp(-1j * k * (xhat[0][:, None] * x[0][None, :] + xhat[1][:, None] * x[1][None, :]))
    return -np.exp(1j * np.pi / 4) / np.sqrt(8 * np.pi * k) * (E * w[None, :]) @ v


def circle_exact_dn(a, k, theta, mmax=None):
    """Exact d_n u on r=a for incidence d=(1,0); theta = polar angle of boundary point."""
    mmax = mmax or int(k * a + 30 + 4 * (k * a) ** (1 / 3))
    m = np.arange(-mmax, mmax + 1)
    Hm = hankel1(m, k * a)
    coef = (1j ** m) * (-2j / (np.pi * a)) / Hm
    return (np.exp(1j * np.outer(theta, m)) * coef[None, :]).sum(1)


def best_origin(geo):
    """Origin c maximising min_x (x-c).n over the boundary (LP); gamma = 0.5 * that minimum."""
    from scipy.optimize import linprog
    x, n = geo["x"], geo["n"]
    # variables (c1, c2, m): maximise m s.t. x.n - c.n >= m
    A = np.stack([n[0], n[1], np.ones_like(n[0])], 1)
    b = (x * n).sum(0)
    res = linprog([0, 0, -1], A_ub=A, b_ub=b, bounds=[(None, None)] * 3)
    return res.x[:2], 0.5 * res.x[2]


def star_combined_at(geo, k, mats, c):
    """Star-combined operator with respect to origin c (any c with (x-c).n > 0)."""
    N = geo["N"]; x = geo["x"] - np.asarray(c)[:, None]
    xn = (x * geo["n"]).sum(0); xt = (x * geo["tau"]).sum(0)
    eta = k * np.hypot(x[0], x[1]) + 0.5j
    A = xn[:, None] * (0.5 * np.eye(N) + mats["Dp"]) + xt[:, None] * mats["T"] - 1j * eta[:, None] * mats["S"]
    return A, eta, 0.5 * xn.min()


def rhs_star_at(geo, k, dirs, eta, c):
    x = geo["x"]; xc = x - np.asarray(c)[:, None]
    ui = np.exp(1j * k * (dirs[0][None, :] * x[0][:, None] + dirs[1][None, :] * x[1][:, None]))
    xd = xc[0][:, None] * dirs[0][None, :] + xc[1][:, None] * dirs[1][None, :]
    return (1j * k * xd - 1j * eta[:, None]) * ui
