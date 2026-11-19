"""Benchmark metadata (shapes, wavenumbers, incident angles) regenerated from the random seeds of the paper
experiments, so that nothing but code has to be uploaded to Colab.
  train   : the training traces of our model (stars, seeds 100 and 150) + support bodies and ellipses (seed 400)
  val     : seeds 200, 250, 500
  test_id : the 24 test stars (seed 300), k = 32..512, 3 incidences each
  test_ood: 12 support-function bodies + 6 ellipses (seed 777), k = 32..512, 3 incidences each"""
import numpy as np
from shapes import random_convex_star, random_support_body
from bie import ellipse

def _stars(seed, nshape, n_k, n_dir, klo, khi):
    rng = np.random.default_rng(seed); out = []
    for i in range(nshape):
        cv = random_convex_star(rng)
        for kk in np.exp(rng.uniform(np.log(klo), np.log(khi), n_k)):
            for al in rng.uniform(0, 2 * np.pi, n_dir):
                out.append(dict(params=cv.params, k=float(kk), alpha=float(al)))
    return out

def _rich(seed, n_sup, n_ell):
    rng = np.random.default_rng(seed)
    shapes = [random_support_body(rng) for _ in range(n_sup)]
    for _ in range(n_ell):
        asp = rng.uniform(1.2, 2.5); b = 1.0 / np.sqrt(asp); shapes.append(ellipse(b * asp, b))
    out = []
    for cv in shapes:
        kk = float(np.exp(rng.uniform(np.log(32), np.log(512))))
        for al in rng.uniform(0, 2 * np.pi, 8):
            out.append(dict(params=cv.params, k=kk, alpha=float(al)))
    return out

def meta(split):
    if split == "train":
        return _stars(100, 400, 2, 6, 32, 128) + _stars(150, 300, 1, 8, 128, 512) + _rich(400, 240, 60)
    if split == "val":
        return _stars(200, 40, 2, 6, 32, 128) + _stars(250, 30, 1, 8, 128, 512) + _rich(500, 24, 6)
    if split == "test_id":
        rng = np.random.default_rng(300); out = []
        shapes = [random_convex_star(rng) for _ in range(24)]
        for cv in shapes:
            al = rng.uniform(0, 2 * np.pi, 3)
            for kk in [32, 64, 128, 256, 512]:
                out += [dict(params=cv.params, k=float(kk), alpha=float(a)) for a in al]
        return out
    rng = np.random.default_rng(777)
    shapes = [random_support_body(rng).params for _ in range(12)]
    shapes += [dict(a=a, b=b) for a, b in [(1.3, 0.72), (1.4, 0.6), (1.5, 0.55), (1.5, 0.5), (1.2, 0.6), (1.35, 0.5)]]
    out = []
    for p in shapes:
        al = rng.uniform(0, 2 * np.pi, 3)
        for kk in [32, 64, 128, 256, 512]:
            out += [dict(params=p, k=float(kk), alpha=float(a)) for a in al]
    return out
