"""Quick start: far field of the co-moving Fock surrogate on a random star-shaped obstacle against a Nystrom reference.
Usage: python examples/quickstart.py [k]   (CPU, a few seconds for k <= 256)"""
import os, sys, time, numpy as np
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"); sys.path.insert(0, SRC)
from shapes import random_convex_star
from bie import solve_density, far_field
from surrogate import Surrogate
from fast_surrogate import surrogate_farfield

k = float(sys.argv[1]) if len(sys.argv) > 1 else 128.0
cv = random_convex_star(np.random.default_rng(2026))                 # unseen shape of the training family
alphas = np.array([0.3, 2.0, 4.1])                                   # incident angles
th = 2 * np.pi * np.arange(1536) / 1536; xh = np.array([np.cos(th), np.sin(th)])
sur = Surrogate(os.path.join(SRC, "g2_mlp_w256_local_ctx_hi.npz"))   # trained for k in [32, 512]
t0 = time.perf_counter(); F, _, _ = surrogate_farfield(sur, cv, k, alphas, xh); ts = time.perf_counter() - t0
g0 = cv.nodes(512); N = int(5.5 * k * g0["sp"].max()) // 2 * 2; geo = cv.nodes(N)
t0 = time.perf_counter()
V = solve_density(geo, k, np.array([np.cos(alphas), np.sin(alphas)]), formulation="cfie")
R = far_field(geo, k, V, xh); tn = time.perf_counter() - t0
for j, a in enumerate(alphas):
    e = np.linalg.norm(F[:, j] - R[:, j]) / np.linalg.norm(R[:, j])
    print(f"k = {k:g}, incident angle {a:.2f}: relative far-field error of the surrogate {e:.1e}")
print(f"time: surrogate {ts:.2f} s, Nystrom ({N} unknowns) {tn:.2f} s")
