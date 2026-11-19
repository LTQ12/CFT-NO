"""Far-field error of several surrogate models on the 24 test stars x 3 incidences of Tables 4-6, k = 32 ... 2048.
Reference: Kress Nystrom CFIE at 5.5 points per wavelength on the GPU (gpu_bie), far field at 1536 directions.
Models: the paper model (trained k <= 512, numpy), the same recipe retrained on the GPU (control), and the
model trained with the k in [512, 2048] extension.
Usage: python wide_eval.py [KS]    -> $MS_OUT/results/wide_eval.json"""
import os, sys, json, time, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")
import gpu_bie, bench_data_colab as BD
from bench_meta import meta
from bie import star_curve
from surrogate import Surrogate
from fast_surrogate import surrogate_farfield
RES = os.path.join(os.environ.get("MS_OUT", ROOT), "results")
KS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "32,64,128,256,512,1024,2048").split(",")]
MODELS = {"paper (k<=512)": os.path.join(ROOT, "g2_mlp_w256_local_ctx_hi.npz"),
          "retrained GPU (k<=512)": os.path.join(RES, "g2_mlp_w256_local_ctx_hi_torch.npz"),
          "wide (k<=2048)": os.path.join(RES, "g2_mlp_w256_local_ctx_hi_xhi_torch.npz")}
rel = lambda F, R: float(np.linalg.norm(F - R) / np.linalg.norm(R))


def main():
    B = BD.gpu_backend(); assert B is not None, "choose a GPU runtime"
    surs = {n: Surrogate(p) for n, p in MODELS.items() if os.path.exists(p)}
    print("models:", list(surs), flush=True)
    S = {}
    for m in meta("test_id"):
        if m["k"] == 512.0:
            S.setdefault(json.dumps(m["params"], sort_keys=True), dict(params=m["params"], alphas=[]))["alphas"].append(m["alpha"])
    S = list(S.values()); rows = []
    for k in KS:
        t0 = time.time()
        for si, sh in enumerate(S):
            p = sh["params"]; cv = star_curve(p["a"], p["b"], p["r0"]); al = np.array(sh["alphas"])
            g0 = cv.nodes(512); N = int(5.5 * k * g0["sp"].max()) // 2 * 2
            Fref = gpu_bie.solve_farfield(B, cv.nodes(N), k, al, BD.TH)
            for name, sur in surs.items():
                F, _, se = surrogate_farfield(sur, cv, k, al, BD.XH)
                rows += [dict(k=k, shape=si, alpha=float(a), method=name, err=rel(F[:, j], Fref[j])) for j, a in enumerate(al)]
            E = se.farfield_operator(BD.XH)
            rows += [dict(k=k, shape=si, alpha=float(a), method="fock",
                          err=rel(E @ se.density(se.charts(a), np.zeros((32, 8), complex)), Fref[j])) for j, a in enumerate(al)]
            if si % 8 == 7:
                print(f"  k={k} shape {si + 1}/{len(S)} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(dict(ks=KS, models=MODELS, rows=rows), open(os.path.join(RES, "wide_eval.json"), "w"))
        print(f"== k = {k}: median (max) relative far-field error", flush=True)
        for meth in list(surs) + ["fock"]:
            e = [r["err"] for r in rows if r["k"] == k and r["method"] == meth]
            print(f"   {meth:24s} {np.median(e):.2e} ({np.max(e):.2e})", flush=True)
    print("saved", os.path.join(RES, "wide_eval.json"), flush=True)


if __name__ == "__main__":
    main()
