"""The Nystrom CFIE solver of bie.py (Kress log-splitting, direct formulation
(1/2 + D') v - i k S v = d_n u_i - i k u_i) written once for two array back ends:
  numpy  (scipy Bessel functions, used to test this file against bie.py),
  torch  (CUDA, torch.special Bessel functions, complex128 LU on the GPU).
Every formula mirrors bie._assemble_rows line by line; nothing is approximated."""
import numpy as np

EULER = 0.57721566490153286
PI = np.pi


def kress_Rrow(N):
    n = N // 2
    tj = 2 * PI * np.arange(N) / N
    m = np.arange(1, n)
    Rrow = np.empty(N)
    for a in range(0, N, 1024):                                    # chunked: N x n cosines
        Rrow[a:a + 1024] = -(2 * PI / n) * (np.cos(np.outer(tj[a:a + 1024], m)) / m).sum(1)
    return Rrow - (PI / n ** 2) * np.cos(n * tj)


class NumpyB:
    name = "numpy"

    def __init__(self):
        from scipy.special import j0, j1, y0, y1
        self.j0, self.j1, self.y0, self.y1 = j0, j1, y0, y1
        self.hypot, self.log, self.sin, self.cos = np.hypot, np.log, np.sin, np.cos

    def f(self, a): return np.asarray(a, np.float64)
    def arange(self, n): return np.arange(n)
    def zeros(self, s): return np.zeros(s)
    def czeros(self, s): return np.zeros(s, np.complex128)
    def c(self, a): return np.asarray(a, np.complex128)
    def cplx(self, re, im): return re + 1j * im
    def solve(self, A, b): return np.linalg.solve(A, b)
    def np(self, a): return np.asarray(a)
    def sync(self): pass


class TorchB:
    def __init__(self, device="cuda", bessel="torch"):
        import torch
        self.t = torch; self.dev = torch.device(device); self.name = f"torch-{device}-{bessel}"
        if bessel == "torch":
            sp = torch.special
            self.j0, self.j1, self.y0, self.y1 = sp.bessel_j0, sp.bessel_j1, sp.bessel_y0, sp.bessel_y1
        else:                                    # cupy's Cephes ports, shared with torch through DLPack
            import cupy, cupyx.scipy.special as cs
            wrap = lambda fn: (lambda x: torch.from_dlpack(fn(cupy.from_dlpack(x.contiguous()))))
            self.j0, self.j1, self.y0, self.y1 = wrap(cs.j0), wrap(cs.j1), wrap(cs.y0), wrap(cs.y1)
        self.hypot, self.log, self.sin, self.cos = torch.hypot, torch.log, torch.sin, torch.cos

    def f(self, a): return self.t.as_tensor(np.asarray(a, np.float64), dtype=self.t.float64, device=self.dev)
    def arange(self, n): return self.t.arange(n, device=self.dev)
    def zeros(self, s): return self.t.zeros(s, dtype=self.t.float64, device=self.dev)
    def czeros(self, s): return self.t.zeros(s, dtype=self.t.complex128, device=self.dev)
    def c(self, a): return self.t.as_tensor(np.asarray(a, np.complex128), dtype=self.t.complex128, device=self.dev)
    def cplx(self, re, im): return self.t.complex(re, im)
    def solve(self, A, b): return self.t.linalg.solve(A, b)
    def np(self, a): return a.detach().cpu().numpy()
    def sync(self):
        if self.dev.type == "cuda": self.t.cuda.synchronize()


def _geo_arrays(B, geo):
    return (B.f(geo["x"][0]), B.f(geo["x"][1]), B.f(geo["n"][0]), B.f(geo["n"][1]),
            B.f(geo["sp"]), B.f(geo["kap"]), B.f(geo["t"]))


def cfie_blocks(B, geo, k, block=1024):
    """Row blocks (i0, i1, A[i0:i1]) of A = 1/2 I + D' - i k S (ds_y included), computed on the back end."""
    N = geo["N"]; w = PI / (N // 2)
    x0, x1, n0, n1, sp, kap, t = _geo_arrays(B, geo)
    Rrow = B.f(kress_Rrow(N)); jj = B.arange(N)
    for i0 in range(0, N, block):
        i1 = min(N, i0 + block); rows = jj[i0:i1]; nb = i1 - i0
        diag = (B.arange(nb), rows)
        dx = x0[rows, None] - x0[None, :]
        dy = x1[rows, None] - x1[None, :]
        r = B.hypot(dx, dy); r[diag] = 1.0
        kr = k * r
        E = B.zeros((nb, N)); E[diag] = 1.0
        L = B.log(4 * B.sin((t[rows, None] - t[None, :]) / 2) ** 2 + E)
        R = Rrow[(jj[None, :] - rows[:, None]) % N]
        spr = sp[rows]
        # single layer
        J, Y = B.j0(kr), B.y0(kr)
        M1 = -(1 / (4 * PI)) * J * sp[None, :]
        M2 = B.cplx(-0.25 * Y * sp[None, :], 0.25 * J * sp[None, :]) - M1 * L        # 0.25i H0 sp - M1 L
        M1[diag] = -(1 / (4 * PI)) * spr
        M2[diag] = B.cplx((-EULER / (2 * PI) - B.log(k * spr / 2) / (2 * PI)) * spr, 0.25 * spr)
        S = R * M1 + w * M2
        del J, Y, M1, M2
        # adjoint double layer
        g = (dx * n0[rows, None] + dy * n1[rows, None]) / r * sp[None, :]
        J, Y = B.j1(kr), B.y1(kr)
        K1 = (k / (4 * PI)) * J * g
        K2 = B.cplx(0.25 * k * Y * g, -0.25 * k * J * g) - K1 * L                   # -0.25i k H1 g - K1 L
        K1[diag] = 0.0
        K2[diag] = B.cplx(-kap[rows] * spr / (4 * PI), 0.0 * spr)
        D = R * K1 + w * K2
        del J, Y, K1, K2, g, dx, dy, r, kr, L, R, E
        blk = D + S * (-1j * k)
        blk[diag] = blk[diag] + 0.5
        del S, D
        yield i0, i1, blk


def cfie_matrix(B, geo, k, block=1024):
    """The full CFIE matrix on the back end."""
    N = geo["N"]; A = B.czeros((N, N))
    for i0, i1, blk in cfie_blocks(B, geo, k, block):
        A[i0:i1] = blk
    x0, x1, n0, n1, sp, _, _ = _geo_arrays(B, geo)
    return A, (x0, x1, n0, n1, sp)


def cfie_apply(B, geo, k, X, block=512):
    """A @ X (numpy, N x m) without storing A: for Galerkin methods that only need the matrix times their basis."""
    Xb = B.c(X); out = B.czeros((geo["N"], X.shape[1]))
    for i0, i1, blk in cfie_blocks(B, geo, k, block):
        out[i0:i1] = blk @ Xb
    B.sync()
    return B.np(out)


def cfie_density(B, geo, k, alphas):
    """Density v (N x len(alphas), numpy) of the CFIE for plane waves with angles alphas."""
    A, (x0, x1, n0, n1, sp) = cfie_matrix(B, geo, k)
    al = B.f(alphas); d0, d1 = B.cos(al), B.sin(al)
    ui = expi(B, k * (x0[:, None] * d0[None, :] + x1[:, None] * d1[None, :]))
    dn = n0[:, None] * d0[None, :] + n1[:, None] * d1[None, :]
    f = ui * B.cplx(0.0 * dn, k * dn) + ui * (-1j * k)
    V = B.solve(A, f); del A
    out = B.np(V); B.sync()
    return out


def expi(B, ph):
    return B.cplx(B.cos(ph), B.sin(ph))


def solve_farfield(B, geo, k, alphas, th):
    """Far field (len(alphas) x len(th)) of the CFIE density, as in bench_data_colab.work."""
    A, (x0, x1, n0, n1, sp) = cfie_matrix(B, geo, k)
    al = B.f(alphas); d0, d1 = B.cos(al), B.sin(al)
    ph = k * (x0[:, None] * d0[None, :] + x1[:, None] * d1[None, :])
    ui = expi(B, ph)
    dn = n0[:, None] * d0[None, :] + n1[:, None] * d1[None, :]
    f = ui * B.cplx(0.0 * dn, k * dn) + ui * (-1j * k)                               # i k (n.d) ui - i k ui
    V = B.solve(A, f)
    del A
    thb = B.f(th)
    w = sp * (2 * PI / geo["N"])
    Eo = expi(B, -k * (B.cos(thb)[:, None] * x0[None, :] + B.sin(thb)[:, None] * x1[None, :]))
    ff = (Eo * B.cplx(w, 0.0 * w)[None, :]) @ V
    ff = ff * complex(-np.exp(1j * PI / 4) / np.sqrt(8 * PI * k))
    out = B.np(ff).T                                                                  # (nalpha, nth)
    B.sync()
    return out


def bessel_selftest(B):
    """Max error of the back end's J0, J1, Y0, Y1 against scipy on [1e-3, 6e3]."""
    from scipy.special import j0, j1, y0, y1
    x = np.concatenate([np.logspace(-3, np.log10(6000), 40001), np.linspace(0.5, 60, 20001)])
    xb = B.f(x); worst = 0.0
    for fb, fs in ((B.j0, j0), (B.j1, j1), (B.y0, y0), (B.y1, y1)):
        ref = fs(x); got = B.np(fb(xb))
        worst = max(worst, float(np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref)))))
    return worst
