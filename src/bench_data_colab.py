"""Generate the neural-operator benchmark on Colab from code only (nothing but .py files is uploaded).
Shapes, wavenumbers and incidences come from the random seeds of the paper experiments (bench_meta.py);
the far fields are computed with the Nystrom CFIE of bie.py, on the GPU (gpu_bie.py, complex128).
Before any data are written the GPU solver is checked against scipy and against bie.py, and every split
is compared with far-field values of the original data (CHECKS below); a split is saved only if all agree
to 1e-5. Without a working GPU the script falls back to bie.py on all CPU cores."""
import json, os, sys, time, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
BENCH_DIR = os.environ.get("BENCH_DIR", "/content/bench_data" if os.path.isdir("/content") else os.path.join(ROOT, "bench_data"))
os.makedirs(BENCH_DIR, exist_ok=True)
from bie import star_curve, ellipse, solve_density, far_field
from shapes import support_body
from bench_meta import meta
import gpu_bie
N_GEO, N_FF = 256, 1536
TH = 2 * np.pi * np.arange(N_FF) / N_FF
XH = np.array([np.cos(TH), np.sin(TH)])

# (sample, direction, Re, Im) of the far fields in the original benchmark files
CHECKS = {
    "train": [(4536, 0, 0.35671359300613403, -0.6529987454414368), (4536, 500, 0.1290580779314041, 0.431167870759964), (4536, 1100, -0.5449623465538025, -0.15295781195163727), (9008, 0, 0.026425989344716072, 0.6877536177635193), (9008, 500, -0.02680927701294422, 0.4822551906108856), (9008, 1100, -0.4412776529788971, 0.46239787340164185)],
    "val": [(462, 0, -0.3649906814098358, -0.47430670261383057), (462, 500, -0.5959805250167847, 0.014012401923537254), (462, 1100, -0.5954615473747253, 0.3437644839286804), (768, 0, 0.060604508966207504, -0.41620007157325745), (768, 500, -0.331939697265625, 0.30264541506767273), (768, 1100, -0.4435233473777771, 0.4750116765499115)],
    "test_id": [(4, 0, 0.156336709856987, 0.677716076374054), (4, 500, 0.42435139417648315, 0.3061875104904175), (4, 1100, -0.33169519901275635, 0.38813546299934387), (244, 0, -0.18657222390174866, 0.5129721164703369), (244, 500, 0.5749875903129578, -0.4803217649459839), (244, 1100, 0.9857376217842102, -0.4675289988517761), (7, 0, 0.6221319437026978, -0.3107755780220032), (7, 500, -0.18961791694164276, -0.48552975058555603), (7, 1100, 0.11195828020572662, 0.4947579503059387)],
    "test_ood": [(113, 0, -0.8373048305511475, -0.09678485989570618), (113, 500, 0.27031514048576355, -0.25723662972450256), (113, 1100, -0.3033718168735504, -0.14462418854236603), (245, 0, 0.3531167209148407, 0.16056497395038605), (245, 500, 0.38610756397247314, 0.3017452359199524), (245, 1100, 0.4952732026576996, 0.08687619119882584), (131, 0, 0.3346390426158905, 0.27952641248703003), (131, 500, -0.458908349275589, -0.14841115474700928), (131, 1100, 0.6065191626548767, -0.42781928181648254)],
}


def shape_of(p):
    if p.get("kind") == "support":
        return support_body(p["c_cos"], p["c_sin"])
    if "r0" in p:
        return star_curve(p["a"], p["b"], p["r0"])
    return ellipse(p["a"], p["b"])


def geometry(cv):
    g = cv.nodes(N_GEO)
    return np.stack([g["x"][0], g["x"][1], g["n"][0], g["n"][1], g["kap"], g["sp"] * 2 * np.pi / N_GEO]).astype(np.float32)


def qvec(p):
    if "r0" in p:
        return np.concatenate([p["a"][1:5], p["b"][1:5]]).astype(np.float32)
    return np.full(8, np.nan, np.float32)


def nodes_for(cv, k):
    g0 = cv.nodes(512); N = int(5.5 * k * g0["sp"].max()) // 2 * 2
    return cv.nodes(N)


def work(group):
    """CPU path (bie.py), used for the fallback and for the self-test."""
    p, k, alphas = group["params"], group["k"], np.array(group["alphas"])
    cv = shape_of(p); geo = nodes_for(cv, k)
    V = solve_density(geo, k, np.array([np.cos(alphas), np.sin(alphas)]), formulation="cfie")
    ff = far_field(geo, k, V, XH).T.astype(np.complex64)
    return geometry(cv), ff, qvec(p)


def work_gpu(B, group):
    p, k, alphas = group["params"], group["k"], np.array(group["alphas"])
    cv = shape_of(p); geo = nodes_for(cv, k)
    ff = gpu_bie.solve_farfield(B, geo, k, alphas, TH).astype(np.complex64)
    return geometry(cv), ff, qvec(p)


def groups_of(split):
    m_all = meta(split); groups = {}
    for i, m in enumerate(m_all):
        key = (json.dumps(m["params"], sort_keys=True), round(float(m["k"]), 9))
        groups.setdefault(key, dict(params=m["params"], k=float(m["k"]), alphas=[], idx=[]))
        groups[key]["alphas"].append(m["alpha"]); groups[key]["idx"].append(i)
    return m_all, sorted(groups.values(), key=lambda g: -g["k"])          # big jobs first


def gpu_backend():
    """torch CUDA back end if it reproduces scipy and bie.py, else None."""
    if os.environ.get("FORCE_CPU") == "1":
        print("FORCE_CPU=1: CPU path", flush=True); return None
    try:
        import torch
        if not torch.cuda.is_available():
            print("no CUDA device: CPU path (choose a GPU runtime to make this fast)", flush=True); return None
        print("GPU", torch.cuda.get_device_name(0), "torch", torch.__version__, flush=True)
        B = None
        for lib in ("torch", "cupy"):
            try:
                Bt = gpu_bie.TorchB("cuda", bessel=lib)
                eb = gpu_bie.bessel_selftest(Bt)
                print(f"self-test 1: {lib} Bessel J0 J1 Y0 Y1 on the GPU vs scipy, max error {eb:.1e}", flush=True)
                if eb < 1e-11:
                    B = Bt; break
            except Exception as ex:
                print(f"self-test 1: {lib} Bessel functions unavailable: {ex!r}", flush=True)
        if B is None:
            print("no accurate Bessel functions on the GPU: CPU path", flush=True); return None
        _, gl = groups_of("test_id")
        g = [x for x in gl if abs(x["k"] - 128) < 1e-6][0]
        t0 = time.time(); _, ff_g, _ = work_gpu(B, g); tg = time.time() - t0
        t0 = time.time(); _, ff_c, _ = work(g); tc = time.time() - t0
        e = np.abs(ff_g - ff_c).max() / np.abs(ff_c).max()
        print(f"self-test 2: GPU solver vs bie.py at k=128, rel diff {e:.1e} (GPU {tg:.1f}s incl. warm-up, CPU {tc:.1f}s)", flush=True)
        if not e < 1e-5:
            print("GPU solver does not reproduce bie.py: CPU path", flush=True); return None
        return B
    except Exception as ex:                                              # any torch problem -> CPU
        print("GPU path unavailable:", repr(ex), "-> CPU path", flush=True); return None


def run(split, B):
    path = os.path.join(BENCH_DIR, f"bench_{split}.npz")
    if os.path.exists(path):
        print(split, "already generated, skipped", flush=True); return
    m_all, glist = groups_of(split)
    n = len(m_all); G = np.zeros((n, 6, N_GEO), np.float32); F = np.zeros((n, N_FF), np.complex64); Q = np.zeros((n, 8), np.float32)
    print(f"{split}: {n} samples, {len(glist)} solves, k = {glist[-1]['k']:.0f}..{glist[0]['k']:.0f}", flush=True)
    t0 = time.time()

    def put(gi, g, res):
        geom, ff, q = res
        for j, i in enumerate(g["idx"]):
            G[i], F[i], Q[i] = geom, ff[j], q
        if gi % 50 == 0 or gi == len(glist) - 1:
            print(f"  {split} {gi + 1}/{len(glist)}  k={g['k']:.0f}  {time.time() - t0:.0f}s", flush=True)

    if B is not None:
        for gi, g in enumerate(glist):
            put(gi, g, work_gpu(B, g))
    else:
        from multiprocessing import Pool
        with Pool(int(os.environ.get("NPROC", str(os.cpu_count())))) as pool:     # ~1.5 GB per worker at k=512
            for gi, (g, res) in enumerate(zip(glist, pool.imap(work, glist))):
                put(gi, g, res)
    for (i, j, re, im) in CHECKS.get(split, []):
        ref = complex(re, im); err = abs(F[i, j] - ref) / max(abs(ref), 1e-12)
        print(f"  check {split} sample {i} dir {j}: rel diff {err:.1e}", flush=True)
        assert err < 1e-5, "regenerated data differ from the original benchmark"
    k = np.array([m["k"] for m in m_all], np.float32); al = np.array([m["alpha"] for m in m_all], np.float32)
    np.savez(path, k=k, alpha=al, geom=G, farfield=F, q=Q)
    print(f"{split} saved to {path}  ({time.time() - t0:.0f}s)", flush=True)


def ensure(splits=("test_id", "test_ood", "val", "train")):
    todo = [s for s in splits if not os.path.exists(os.path.join(BENCH_DIR, f"bench_{s}.npz"))]
    if not todo:
        return
    B = gpu_backend()
    for sp in todo:
        run(sp, B)


if __name__ == "__main__":
    ensure(sys.argv[1:] or ["test_id", "test_ood", "val", "train"])
