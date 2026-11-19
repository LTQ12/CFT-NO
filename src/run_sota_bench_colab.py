"""Colab GPU runner: generic neural-operator baselines on the high-frequency convex benchmark.
Task: boundary geometry (256 points) + incidence + k  ->  far field at 1536 directions (k = 32..512).
Models: official neuraloperator FNO, TFNO, GINO; DeepONet (plain torch) with a standard trunk and with a
multiscale Fourier-feature trunk (the spectral-bias remedy of Mscale-DeepONet).
Each model is trained twice: with geometry only ("plain") and with the incident field on the boundary
added as two input channels ("phase"). Output: sota_results.json (per-sample test errors)."""
import json, os, time, math, numpy as np, torch, torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = int(os.environ.get("EPOCHS", "200")); BATCH = int(os.environ.get("BATCH", "32"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]
MODELS = os.environ.get("MODELS", "FNO,TFNO,DeepONet,DeepONetFourier").split(",")
VARIANTS = os.environ.get("VARIANTS", "plain,phase").split(",")
N_FF = 1536
KMAX = float(os.environ.get("KMAX", "1e9"))      # frequency extrapolation: train/validate only on k <= KMAX
BENCH_DIR = os.environ.get("BENCH_DIR", "/content/bench_data" if os.path.isdir("/content") else os.path.join(ROOT, "bench_data"))
os.environ["BENCH_DIR"] = BENCH_DIR
RES_DIR = os.path.join(os.environ.get("MS_OUT", ROOT), "results"); os.makedirs(RES_DIR, exist_ok=True)
OUT = os.environ.get("OUT", "sota_results.json" if KMAX > 1e8 else f"sota_results_kmax{int(KMAX)}.json")

def load(split, kmax=None):
    z = np.load(os.path.join(BENCH_DIR, f"bench_{split}.npz"))
    d = {k: z[k] for k in z.files}
    if kmax is not None:
        keep = d["k"] <= kmax * (1 + 1e-6)
        d = {k: v[keep] for k, v in d.items()}
    return d

def inputs(pack, variant):
    g = pack["geom"]; k = pack["k"][:, None]; al = pack["alpha"][:, None]
    x, y, nx, ny, kap, w = [g[:, i] for i in range(6)]
    ones = np.ones_like(x)
    ch = [x, y, nx, ny, kap / 5, w * 256 / (2 * np.pi), np.cos(al) * ones, np.sin(al) * ones,
          np.log(k / 64) * ones, (32 / k) * ones]
    if variant == "phase":
        ph = k * (np.cos(al) * x + np.sin(al) * y)
        ch += [np.cos(ph), np.sin(ph)]
    return np.stack(ch, 1).astype(np.float32)            # (n, C, 256)

def targets(pack, p):
    f = pack["farfield"] / (pack["k"][:, None] ** p)
    return np.stack([f.real, f.imag], 1).astype(np.float32)   # (n, 2, 1536)

class FNOWrap(nn.Module):
    def __init__(self, C, kind):
        super().__init__()
        from neuralop.models import FNO, TFNO
        kw = dict(n_modes=(64,), hidden_channels=96, in_channels=C, out_channels=2, n_layers=4)
        if kind == "TFNO":
            kw.update(factorization="tucker", implementation="factorized", rank=0.2)
        self.m = (FNO if kind == "FNO" else TFNO)(**kw)
    def forward(self, x):
        try:
            y = self.m(x, output_shape=(N_FF,))
        except TypeError:                                  # older/newer neuralop without output_shape
            y = self.m(x)
        if y.shape[-1] != N_FF:
            y = nn.functional.interpolate(y, size=N_FF, mode="linear", align_corners=False)
        return y

class GINOWrap(nn.Module):
    def __init__(self, C):
        super().__init__()
        from neuralop.models import GINO
        self.m = GINO(in_channels=C, out_channels=2, gno_coord_dim=2, in_gno_radius=0.06, out_gno_radius=0.06,
                      gno_use_open3d=False, gno_use_torch_scatter=False,
                      in_gno_transform_type="linear", out_gno_transform_type="linear",
                      in_gno_channel_mlp_hidden_layers=[64, 64], out_gno_channel_mlp_hidden_layers=[64, 64],
                      fno_n_modes=(16, 16), fno_in_channels=C, fno_hidden_channels=64, fno_n_layers=4,
                      fno_lifting_channel_ratio=2, projection_channel_ratio=2)
        ax = torch.linspace(0, 1, 32); self.register_buffer("lat", torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1))
        th = torch.arange(N_FF) * 2 * math.pi / N_FF
        self.register_buffer("q", torch.stack([0.5 + 0.45 * torch.cos(th), 0.5 + 0.45 * torch.sin(th)], -1))
    def forward(self, x):
        out = []
        for b in range(x.shape[0]):                       # GINO takes one geometry at a time
            pts = (x[b, :2].T / 3.2 + 0.5)
            y = self.m(x=x[b:b + 1].transpose(1, 2), input_geom=pts, latent_queries=self.lat, output_queries=self.q)
            out.append(y.transpose(1, 2))
        return torch.cat(out, 0)

class DeepONet(nn.Module):
    def __init__(self, C, fourier, width=256, p=256):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(C * 256, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                    nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2 * p))
        th = torch.arange(N_FF) * 2 * math.pi / N_FF
        if fourier:                                       # multiscale Fourier trunk features up to angular frequency 768
            m = torch.cat([torch.arange(1, 64), torch.linspace(64, 768, 128).round()])
            feats = torch.cat([torch.cos(th[:, None] * m), torch.sin(th[:, None] * m)], 1)
        else:
            feats = torch.stack([torch.cos(th), torch.sin(th)], 1)
        self.register_buffer("tf", feats)
        self.trunk = nn.Sequential(nn.Linear(feats.shape[1], width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                   nn.Linear(width, p))
        self.p = p
    def forward(self, x):
        b = self.branch(x.flatten(1)).view(x.shape[0], 2, self.p)
        t = self.trunk(self.tf)                            # (N_FF, p)
        return torch.einsum("bcp,np->bcn", b, t) / math.sqrt(self.p)

def build(name, C):
    if name in ("FNO", "TFNO"):
        return FNOWrap(C, name)
    if name == "GINO":
        return GINOWrap(C)
    return DeepONet(C, fourier=(name == "DeepONetFourier"))

def rel(pred, tgt):
    return ((pred - tgt) ** 2).sum((1, 2)).sqrt() / (tgt ** 2).sum((1, 2)).sqrt().clamp_min(1e-12)

def save(results):
    """Write the results to Drive and to /content, check that each file is really there."""
    for d in (RES_DIR, ROOT):
        f = os.path.join(d, OUT)
        try:
            with open(f, "w") as fh:
                json.dump(results, fh, default=float); fh.flush(); os.fsync(fh.fileno())
            print(f"  saved {f} ({os.path.getsize(f) / 1e3:.0f} kB)", flush=True)
        except Exception as e:
            print(f"  SAVE FAILED {f}: {e!r}", flush=True)

def summary(rec, test_k):
    """Median error per k and per shape family, printed so that the log alone carries the table."""
    for sp in ("test_id", "test_ood"):
        k = np.round(np.array(test_k[sp])).astype(int); ks = sorted(int(x) for x in set(k))
        e = np.array(rec[sp + "_per_sample"])
        if sp == "test_id":
            groups = [("star", np.ones(len(k), bool))]
        else:                                    # first 12 shapes (15 samples each): support bodies, then 6 ellipses
            sh = np.arange(len(k)) // 15
            groups = [("support", sh < 12), ("ellipse", sh >= 12)]
        for g, m in groups:
            row = " ".join(f"{np.median(e[m & (k == kk)]):.3g}" for kk in ks)
            print(f"  TABLE {rec['variant']} {rec['model']} {sp} {g} k={ks}: {row} | all {np.median(e[m]):.3g}", flush=True)
    print(f"  TABLE {rec['variant']} {rec['model']} params {rec['n_params']} train {rec['train_time_s'] / 60:.1f} min "
          f"infer {rec['test_id_time_per_sample_s'] * 1e3:.2f} ms/sample", flush=True)

def main():
    import sys; sys.path.insert(0, ROOT)
    import bench_data_colab; bench_data_colab.ensure()       # data missing (e.g. after a disconnect): regenerate on the GPU
    print("device", DEV, torch.cuda.get_device_name(0) if DEV.type == "cuda" else "", flush=True)
    tr, va = load("train", KMAX), load("val", KMAX); print("train samples", len(tr["k"]), "k max %.0f" % tr["k"].max(), flush=True)
    tests = {s: load(s) for s in ["test_id", "test_ood"]}
    nrm = np.sqrt((abs(tr["farfield"]) ** 2).mean(1)); p = float(np.polyfit(np.log(tr["k"]), np.log(nrm), 1)[0])
    results = dict(p=p, kmax=KMAX, test_k={sp: pk["k"].tolist() for sp, pk in tests.items()}, torch=torch.__version__, device=str(DEV), runs=[])
    try:
        import neuralop; results["neuralop"] = neuralop.__version__
    except Exception as e:
        results["neuralop"] = repr(e)
    for variant in VARIANTS:
        Xtr, Ytr = torch.tensor(inputs(tr, variant)), torch.tensor(targets(tr, p))
        mu, sd = Xtr.mean((0, 2), keepdim=True), Xtr.std((0, 2), keepdim=True).clamp_min(1e-5)
        prep = lambda pack: ((torch.tensor(inputs(pack, variant)) - mu) / sd, torch.tensor(targets(pack, p)))
        Xtr = (Xtr - mu) / sd
        for name in MODELS:
            for seed in SEEDS:
              try:
                torch.manual_seed(seed); np.random.seed(seed)
                try:
                    model = build(name, Xtr.shape[1]).to(DEV)
                except Exception as e:
                    print("skip", name, repr(e), flush=True); results["runs"].append(dict(model=name, variant=variant, error=repr(e))); continue
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
                epochs = int(os.environ.get("GINO_EPOCHS", "40")) if name == "GINO" else EPOCHS   # GINO runs one geometry at a time
                steps = epochs * math.ceil(len(Xtr) / BATCH); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
                t0 = time.time()
                for ep in range(epochs):
                    perm = torch.randperm(len(Xtr)); tot = 0
                    model.train()
                    for i in range(0, len(Xtr), BATCH):
                        idx = perm[i:i + BATCH]
                        xb, yb = Xtr[idx].to(DEV), Ytr[idx].to(DEV)
                        loss = rel(model(xb), yb).mean()
                        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(idx)
                    if ep % 20 == 0 or ep == epochs - 1:
                        print(variant, name, seed, ep, "train rel %.4f" % (tot / len(Xtr)), f"{time.time() - t0:.0f}s", flush=True)
                ttrain = time.time() - t0
                model.eval(); rec = dict(model=name, variant=variant, seed=seed, train_time_s=ttrain,
                                         n_params=sum(q.numel() for q in model.parameters()))
                with torch.no_grad():
                    for split, pack in [("val", va)] + list(tests.items()):
                        X, Y = prep(pack); errs = []
                        if DEV.type == "cuda": torch.cuda.synchronize()
                        t1 = time.time()
                        for i in range(0, len(X), 64):
                            errs.append(rel(model(X[i:i + 64].to(DEV)), Y[i:i + 64].to(DEV)).cpu())
                        if DEV.type == "cuda": torch.cuda.synchronize()
                        rec[f"{split}_per_sample"] = torch.cat(errs).tolist()
                        rec[f"{split}_time_per_sample_s"] = (time.time() - t1) / len(X)
                        print(variant, name, seed, split, "median rel %.4f" % float(torch.cat(errs).median()), flush=True)
                results["runs"].append(rec)
                summary(rec, results["test_k"])
                save(results)
              except Exception as e:
                import traceback; traceback.print_exc()
                results["runs"].append(dict(model=name, variant=variant, seed=seed, error=repr(e)))
                save(results)
    print("ALL DONE")

if __name__ == "__main__":
    main()
