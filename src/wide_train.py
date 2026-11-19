"""Train the chart-local network of the paper (variant local_ctx_hi: 45 inputs, 3 x 256 tanh, 16 outputs) on the GPU.
Same recipe as g2_train.py: targets = chart coefficients scaled by k^p sqrt(chart width), p fitted on the
k in [32,128] split, inputs standardised, Adam, cosine learning rate 2e-3, batch 512, 150 epochs.
EXTRA=xhi adds the k in [512, 2048] traces (train_xhi / val_xhi). The model is written in the npz format of
mlp.MLP, so surrogate.py and fast_surrogate.py load it unchanged.
Usage: python wide_train.py VARIANT [EPOCHS]   VARIANT = local_ctx_hi_torch  or  local_ctx_hi_xhi_torch"""
import os, sys, time, json, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import torch
from g2_data import add_context
WIDE_DIR = os.environ.get("WIDE_DIR", "/content/wide_data")
RES = os.path.join(os.environ.get("MS_OUT", ROOT), "results"); os.makedirs(RES, exist_ok=True)
variant = sys.argv[1]; epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 150
flags = variant.split("_"); SEED = int(os.environ.get("SEED", "0"))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("device", dev, flush=True)

L = lambda s: np.load(os.path.join(WIDE_DIR, f"g2_{s}.npz"))
tr, va = L("train"), L("val")
r = np.sqrt((abs(tr["Y"]) ** 2 * tr["W"][:, :, None]).sum((1, 2)))
p = -np.polyfit(np.log(tr["k"]), np.log(r), 1)[0]
print("rms(Y sqrt W) ~ k^-%.3f" % p, flush=True)
trs, vas = ["train", "train_hi"], ["val", "val_hi"]
if "xhi" in flags:
    trs.append("train_xhi"); vas.append("val_xhi")
def cat(names):
    D = [L(s) for s in names]
    return {f: np.concatenate([d[f] for d in D]) for f in ("X", "Y", "W", "k")}
T, V = cat(trs), cat(vas)
print("train traces", {s: len(L(s)["k"]) for s in trs}, "val", {s: len(L(s)["k"]) for s in vas}, flush=True)
prep = lambda X: add_context(X[..., :-13])                       # local features + neighbour-chart context
def scaled(D):
    Z = D["Y"] * (D["k"][:, None, None] ** p) * np.sqrt(D["W"])[:, :, None]
    return np.concatenate([Z.real, Z.imag], -1).reshape(-1, 16)
XT, XV = prep(T["X"]), prep(V["X"])
Xtr, Xva = XT.reshape(-1, XT.shape[-1]), XV.reshape(-1, XV.shape[-1])
Ytr, Yva = scaled(T), scaled(V)
xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8; ys = np.sqrt((Ytr ** 2).mean())
g = lambda a: torch.as_tensor(a.astype(np.float32), device=dev)
Xn, Yn, Xvn, Yvn = g((Xtr - xm) / xs), g(Ytr / ys), g((Xva - xm) / xs), g(Yva / ys)

sizes = [Xn.shape[1], 256, 256, 256, 16]
rng = np.random.default_rng(SEED)                                # same initialisation as mlp.MLP
Ws = [torch.nn.Parameter(g(rng.standard_normal((a, b)) * np.sqrt(1.0 / a))) for a, b in zip(sizes[:-1], sizes[1:])]
bs_ = [torch.nn.Parameter(torch.zeros(b, device=dev)) for b in sizes[1:]]
def fwd(X):
    h = X
    for i, (W, b) in enumerate(zip(Ws, bs_)):
        h = h @ W + b
        if i < len(Ws) - 1:
            h = torch.tanh(h)
    return h
n = len(Xn); bs = 512; steps = epochs * ((n + bs - 1) // bs)
opt = torch.optim.Adam(Ws + bs_, lr=2e-3, betas=(0.9, 0.999), eps=1e-8)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + np.cos(np.pi * min(t + 1, steps) / steps)))
t0 = time.time(); hist = []
for ep in range(epochs):
    perm = torch.as_tensor(rng.permutation(n), device=dev); tot = 0.0
    for i in range(0, n, bs):
        idx = perm[i:i + bs]
        loss = ((fwd(Xn[idx]) - Yn[idx]) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(idx)
    if ep % 10 == 0 or ep == epochs - 1:
        with torch.no_grad():
            vl = float(((fwd(Xvn) - Yvn) ** 2).sum(1).mean())
        hist.append(dict(epoch=ep, train=tot / n, val=vl))
        print(f"  {variant} epoch {ep} train {tot / n:.4e} val {vl:.4e} ({time.time() - t0:.0f}s)", flush=True)
with torch.no_grad():
    P = fwd(Xvn).cpu().numpy() * ys
Tt = Yvn.cpu().numpy() * ys
P, Tt = P.reshape(len(V["k"]), -1, 16), Tt.reshape(len(V["k"]), -1, 16)
rel = np.sqrt(((P - Tt) ** 2).sum((1, 2)) / (Tt ** 2).sum((1, 2)))
for lo, hi in ((32, 128), (128, 512), (512, 2049)):
    m = (V["k"] >= lo) & (V["k"] < hi)
    if m.any():
        print(f"  val k in [{lo},{min(hi, 2048)}]: median relative error of the correction {np.median(rel[m]):.4f}", flush=True)
out = os.path.join(RES, f"g2_mlp_w256_{variant}.npz")
Wn = [W.detach().cpu().numpy() for W in Ws]; bn = [b.detach().cpu().numpy() for b in bs_]
np.savez(out, *Wn, *bn, n=len(Wn), xm=xm, xs=xs, p=p, ys=ys, variant=variant)
json.dump(dict(p=float(p), hist=hist, train_time_s=time.time() - t0, val_rel_median=float(np.median(rel))),
          open(os.path.join(RES, f"g2_train_w256_{variant}.json"), "w"))
print("saved", out, flush=True)
