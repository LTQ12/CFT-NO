"""Shape families: smooth convex star-shaped curves and ellipses."""
import numpy as np
from bie import star_curve, ellipse


def random_convex_star(rng, mmax=5, amp=0.12, kmin=0.25, name="star"):
    while True:
        m = np.arange(1, mmax + 1)
        a = amp * rng.uniform(-1, 1, mmax) / m ** 1.5
        b = amp * rng.uniform(-1, 1, mmax) / m ** 1.5
        a[0] = b[0] = 0.0                       # keep origin near the centre
        cv = star_curve(a, b, 1.0, name)
        g = cv.nodes(1024)
        if g["kap"].min() > kmin and ((g["x"] * g["n"]).sum(0)).min() > 0.3:
            return cv


def g1_shapes(seed=0):
    rng = np.random.default_rng(seed)
    shapes = [random_convex_star(rng, name=f"star{i}") for i in range(16)]
    for i, (a, b) in enumerate([(1.0, 0.6), (1.2, 0.7), (1.0, 0.5), (0.9, 0.75)]):
        shapes.append(ellipse(a, b, name=f"ellipse{i}"))
    return shapes


def support_body(c_cos, c_sin, name="support"):
    """Smooth convex body from its support function h(theta) = 1 + sum_m c_m cos(m theta) + s_m sin(m theta)
    (m = 2, 3, ...). Boundary parametrised by the outward-normal angle theta:
    z = h n + h' n_perp, z' = rho n_perp with rho = h + h'' > 0 the radius of curvature."""
    from bie import Curve
    cc = np.asarray(c_cos, float); cs = np.asarray(c_sin, float)
    m = np.arange(2, 2 + len(cc))

    def h(t, d=0):
        C, S = np.cos(np.outer(t, m)), np.sin(np.outer(t, m))
        if d == 0:
            return 1 + C @ cc + S @ cs
        if d == 1:
            return -(S * m) @ cc + (C * m) @ cs
        if d == 2:
            return -(C * m ** 2) @ cc - (S * m ** 2) @ cs
        return (S * m ** 3) @ cc - (C * m ** 3) @ cs

    def z(t):
        n = np.array([np.cos(t), np.sin(t)]); p = np.array([-np.sin(t), np.cos(t)])
        return h(t) * n + h(t, 1) * p

    def dz(t):
        rho = h(t) + h(t, 2); return rho * np.array([-np.sin(t), np.cos(t)])

    def ddz(t):
        rho = h(t) + h(t, 2); rho1 = h(t, 1) + h(t, 3)
        return rho1 * np.array([-np.sin(t), np.cos(t)]) - rho * np.array([np.cos(t), np.sin(t)])

    cv = Curve(z, dz, ddz, name)
    cv.params = dict(kind="support", c_cos=cc.tolist(), c_sin=cs.tolist())
    return cv


def random_support_body(rng, mmax=8, amp=0.35, amp2=0.25, rho_min=0.15, rho_max_ratio=15.0, name="support"):
    """Strongly non-circular smooth convex bodies (outside the training family): elongation from the
    second mode (|c_2| up to amp2), irregularity from modes 3..mmax."""
    while True:
        m = np.arange(2, mmax + 1)
        cc = amp * rng.uniform(-1, 1, len(m)) / m ** 2
        cs = amp * rng.uniform(-1, 1, len(m)) / m ** 2
        cc[0], cs[0] = amp2 * rng.uniform(-1, 1), amp2 * rng.uniform(-1, 1)
        cv = support_body(cc, cs, name)
        t = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
        rho = np.hypot(*cv.dz(t))
        hmin = (np.array([np.cos(t), np.sin(t)]) * cv.z(t)).sum(0).min()
        if rho.min() > rho_min and rho.max() / rho.min() < rho_max_ratio and hmin > 0.3:
            return cv
