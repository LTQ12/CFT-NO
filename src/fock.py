"""Universal Fock current for a sound-soft smooth convex boundary, tabulated from the exact
series solution on a large circle (ka -> infinity limit, checked by Richardson-type comparison).

Fock variable: xi = (k/2)^(1/3) * int_{s_sb}^{s} rho^(-2/3) ds   (positive into the shadow)
Lit side   : v ~ e^{ik d.x} k (2/(k rho))^(1/3) F(xi),  F(xi) -> -2 i xi  as xi -> -inf  (PO)
Shadow side: v ~ e^{ik phi_c} k (2/(k rho))^(1/3) G(xi), phi_c = creeping phase, G = F e^{ik(d.x - phi_c)}
"""
import numpy as np
from scipy.special import hankel1


def circle_trace_near_shadow(ka, sig, a=1.0):
    """Exact d_n u on a circle of radius a, k = ka/a, d=(1,0), at signed arclength sig from the
    shadow point (sig>0 into the shadow). Returns v and the incident phase d.x."""
    k = ka / a
    theta = np.pi / 2 - sig / a
    mmax = int(ka + 40 + 6 * ka ** (1 / 3))
    m = np.arange(-mmax, mmax + 1)
    coef = (1j ** m) * (-2j / (np.pi * a)) / hankel1(m, ka)
    v = np.zeros(len(theta), complex)
    for i in range(0, len(theta), 200):
        v[i:i + 200] = np.exp(1j * np.outer(theta[i:i + 200], m)) @ coef
    return v, a * np.cos(theta)


def tabulate(ka=4000.0, xi_min=-24.0, xi_max=14.0, n=7601):
    a = 1.0; k = ka
    xi = np.linspace(xi_min, xi_max, n)
    sig = xi * (2 * a ** 2 / k) ** (1 / 3)
    v, dx = circle_trace_near_shadow(ka, sig, a)
    scale = k * (2 / (k * a)) ** (1 / 3)
    theta = np.pi / 2 - sig / a
    dn = np.cos(theta)                                 # d.n on the circle, <0 lit
    vpo = np.where(dn < 0, 2j * k * dn * np.exp(1j * k * dx), 0)
    F = (v - vpo) * np.exp(-1j * k * dx) / scale       # Fock correction to PO (lit and shadow)
    # creeping phase measured from the shadow point: phi_c = d.x_sb + sig  (d.x_sb = 0 here)
    G = v * np.exp(-1j * k * sig) / scale
    return xi, F, G


if __name__ == "__main__":
    import json
    res = {}
    tabs = {ka: tabulate(ka) for ka in [1000.0, 4000.0, 16000.0]}
    xi = tabs[4000.0][0]
    for ka in [1000.0, 4000.0]:
        dF = np.abs(tabs[ka][1] - tabs[16000.0][1]).max()
        res[f"maxdiff_F_ka{int(ka)}_vs_16000"] = float(dF)
    print(res)
    np.savez("results/fock_table.npz", xi=xi, F=tabs[16000.0][1], G=tabs[16000.0][2],
             F4000=tabs[4000.0][1])
    i0 = np.argmin(abs(xi)); print("F(0) =", tabs[16000.0][1][i0], " F(-12)+2i*xi:", tabs[16000.0][1][0] + 2j * xi[0],
          " |G(8)|:", abs(tabs[16000.0][2][np.argmin(abs(xi - 8))]))


class FockBase:
    """Analytic base density: PO plus Fock corrections from the two shadow-boundary points,
    with Fock variable xi = (k/2)^(1/3) int rho^(-2/3) ds and amplitude k (2/(k rho(x)))^(1/3)."""

    def __init__(self, table="results/fock_table.npz"):
        z = np.load(table)
        self.xi, self.F, self.G = z["xi"], z["F"], z["G"]
        sel = self.xi < self.xi[0] + 4
        self.p_tail = -np.polyfit(np.log(-self.xi[sel]), np.log(np.abs(self.F[sel])), 1)[0]

    def _interp(self, tab, x):
        from scipy.interpolate import CubicSpline
        key = id(tab)
        if not hasattr(self, "_sp"):
            self._sp = {}
        if key not in self._sp:
            self._sp[key] = CubicSpline(self.xi, tab)
        out = np.zeros(len(x), complex)
        inside = (x >= self.xi[0]) & (x <= self.xi[-1])
        out[inside] = self._sp[key](x[inside])
        # lit tail beyond the table: algebraic decay fitted on the last 4 units of xi
        lo = x < self.xi[0]
        if lo.any() and tab is self.F:
            out[lo] = tab[0] * (self.xi[0] / x[lo]) ** self.p_tail
        return out

    def density(self, ag, s, k, d, s_in, s_out, parts=("po", "fock")):
        from charts import ArcGeom
        if not hasattr(ag, "I_f"):
            _, _, _, kap_f = ag.point(ag.s_f)
            g = np.abs(kap_f) ** (2 / 3)
            ag.I_f = np.concatenate([[0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(ag.s_f))])
        I = lambda q: np.interp(np.mod(q, ag.L), ag.s_f, ag.I_f)
        Itot = ag.I_f[-1]
        wrap = lambda q: (q + Itot / 2) % Itot - Itot / 2
        _, x, n, kap = ag.point(s)
        dx = d @ x; dn = d @ n
        out = np.zeros(len(s), complex)
        if "po" in parts:
            out += np.where(dn < 0, 2j * k * dn * np.exp(1j * k * dx), 0)
        if "fock" in parts:
            scale = k * (2 * kap / k) ** (1 / 3)
            c = (k / 2) ** (1 / 3)
            _, xb, _, _ = ag.point(np.array([s_in, s_out]))
            Ls = (s_out - s_in) % ag.L
            in_shadow = ((s - s_in) % ag.L) < Ls
            Iin, Iout, Is = I(s_in), I(s_out), I(s)
            branches = [  # (xi on the shadow arc (>0), xi on the lit arc (<0), creeping phase)
                (c * ((Is - Iin) % Itot), -c * ((Iin - Is) % Itot), d @ xb[:, 0] + (s - s_in) % ag.L),
                (c * ((Iout - Is) % Itot), -c * ((Is - Iout) % Itot), d @ xb[:, 1] + (s_out - s) % ag.L)]
            for xi_sh, xi_lit, phc in branches:
                term = np.where(in_shadow, self._interp(self.G, np.where(in_shadow, xi_sh, 0)) * np.exp(1j * k * phc),
                                self._interp(self.F, np.where(in_shadow, -1.0, xi_lit)) * np.exp(1j * k * dx))
                out += scale * term
        return out
