"""Small numpy MLP (tanh, Adam, cosine schedule) used because no deep-learning framework is
installable in this environment. Shared across charts: one row = one chart."""
import numpy as np


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.standard_normal((a, b)).astype(np.float32) * np.sqrt(1.0 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]

    def forward(self, X, keep=False):
        hs = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.tanh(z) if i < len(self.W) - 1 else z
            hs.append(h)
        return (h, hs) if keep else h

    def grads(self, X, Y, wrow=None):
        P, hs = self.forward(X, keep=True)
        R = P - Y
        if wrow is not None:
            R = R * wrow[:, None]
        loss = float((R * (P - Y)).sum() / len(X))
        g = 2 * R / len(X)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = hs[i].T @ g
            gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i].T) * (1 - hs[i] ** 2)
        return loss, gW, gb

    def fit(self, X, Y, Xv=None, Yv=None, epochs=100, bs=1024, lr=2e-3, wd=0.0, seed=0, log=print, wrow=None):
        rng = np.random.default_rng(seed)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        n = len(X); steps = epochs * ((n + bs - 1) // bs)
        hist = []
        for ep in range(epochs):
            perm = rng.permutation(n); tot = 0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                loss, gW, gb = self.grads(X[idx], Y[idx], None if wrow is None else wrow[idx])
                tot += loss * len(idx)
                t += 1
                lr_t = lr * 0.5 * (1 + np.cos(np.pi * t / steps))
                for j, (p, g) in enumerate(zip(params, gW + gb)):
                    if wd and j < len(self.W):
                        g = g + wd * p
                    m[j] = b1 * m[j] + (1 - b1) * g; v[j] = b2 * v[j] + (1 - b2) * g * g
                    p -= lr_t * (m[j] / (1 - b1 ** t)) / (np.sqrt(v[j] / (1 - b2 ** t)) + eps)
            rec = dict(epoch=ep, train=tot / n)
            if Xv is not None:
                rec["val"] = float(((self.forward(Xv) - Yv) ** 2).sum(1).mean())
            hist.append(rec)
            if ep % 10 == 0 or ep == epochs - 1:
                log(rec)
        return hist

    def save(self, path, **extra):
        np.savez(path, *self.W, *self.b, n=len(self.W), **extra)

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=True)
        if "order" in z.files:
            from order_net import OrderNet
            return OrderNet.from_npz(z), z
        n = int(z["n"]); m = MLP.__new__(MLP)
        m.W = [z[f"arr_{i}"] for i in range(n)]; m.b = [z[f"arr_{i + n}"] for i in range(n)]
        return m, z
