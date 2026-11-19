"""Beyond the training band: the surrogate (trained for k <= 512) at k = 512, 1024, 2048 on the 24 test stars of
Tables 4-6 (same shapes and incident directions, bench_meta test_id), against
  reference : Kress Nystrom CFIE at 5.5 points per wavelength, solved on the GPU (gpu_bie, complex128),
              far field at 1536 directions (the protocol of Tables 4-6);
  Fock      : the Fock current alone (the surrogate's base, network off);
  PO        : physical optics;
  EE18/EO17 : the Galerkin hybrid numerical-asymptotic methods of Ecevit-Eruslu and Ecevit-Ozen (hna_galerkin.py,
              unchanged), whose Galerkin matrix B^H W A B needs only A @ B: computed on the GPU without storing A.
Checks before any result is written: GPU Bessel functions and GPU solver against scipy/bie.py (bench_data_colab),
the k = 512 reference against a far-field value of the original benchmark, and at the largest k one shape at
5.5 and 7 points per wavelength.
Usage: python band_eval.py [KS] [HNA]   e.g.  python band_eval.py 512,1024,2048 1
Output: $MS_OUT/results/band_eval.json (rows per trace and method), progress printed line by line."""
import os, sys, json, time, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")
import gpu_bie
from bench_meta import meta
import bench_data_colab as BD
from bie import star_curve, far_field
from surrogate import Surrogate
from fast_surrogate import surrogate_farfield
import hna_galerkin as H

RES = os.path.join(os.environ.get("MS_OUT", ROOT), "results"); os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, os.environ.get("OUT", "band_eval.json"))
KS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "512,1024,2048").split(",")]
DO_HNA = (sys.argv[2] if len(sys.argv) > 2 else "1") == "1"
MODEL = os.path.join(ROOT, "g2_mlp_w256_local_ctx_hi.npz")
XH = BD.XH
EE18_RATIOS = (2.0, 2.0)                       # tuned on the unit circle, as in the paper's Table 5
EE18_D, EO17_D = (8, 12, 16, 20), (8, 12, 16)


def rel(F, ref):
    return float(np.linalg.norm(F - ref) / np.linalg.norm(ref))


def shapes():
    """The 24 test stars with their three incident directions (bench test_id at k = 512)."""
    out = {}
    for m in meta("test_id"):
        if m["k"] == 512.0:
            key = json.dumps(m["params"], sort_keys=True)
            out.setdefault(key, dict(params=m["params"], alphas=[]))["alphas"].append(m["alpha"])
    return list(out.values())


def reference(B, cv, k, al, ppw=5.5):
    g0 = cv.nodes(512); N = int(ppw * k * g0["sp"].max()) // 2 * 2
    t0 = time.time(); F = gpu_bie.solve_farfield(B, cv.nodes(N), k, np.asarray(al), BD.TH); return F, N, time.time() - t0


def hna(B, cv, k, al, ref):
    """EE18 and EO17 at several polynomial degrees for one incidence; A @ [all bases] in one GPU pass."""
    b = H.Boundary(cv, k, al)
    x, n = b.geo["x"], b.geo["n"]; ui = np.exp(1j * k * (b.al @ x)); f = 1j * k * (b.al @ n) * ui - 1j * k * ui
    P = H.default_params(b, *EE18_RATIOS)
    bases = []
    for d in EE18_D:
        I = H.ee18_intervals(b, k, *P, 6)
        bases.append((f"ee18_d{d}", H.basis(b, k, I, d, lambda kind, s, a, bb: H.ee18_phi(kind, s, a, bb, k, b, P))))
    for d in EO17_D:
        xi1, xi2 = b.s_spec - b.t1, b.t2 - b.s_spec; z1, z2 = b.t1, b.L - b.t2
        bases.append((f"eo17_d{d}", H.basis(b, k, H.eo17_intervals(b, k, 2, xi1, z1, xi2, z2), d, None)))
    Bcat = np.concatenate([Bm for _, Bm in bases], 1)
    t0 = time.time(); AB = gpu_bie.cfie_apply(B, b.geo, k, Bcat); t_apply = time.time() - t0
    rows, c0 = [], 0
    for name, Bm in bases:
        m = Bm.shape[1]; ABm = AB[:, c0:c0 + m]; c0 += m
        BW = Bm.conj().T * b.w[None, :]
        c = np.linalg.lstsq(BW @ ABm, BW @ f, rcond=None)[0]
        F = far_field(b.geo, k, (Bm @ c)[:, None], XH)[:, 0]
        rows.append(dict(method=name, err=rel(F, ref), dof=m, nystrom_nodes=b.N))
    return rows, t_apply


def main():
    B = gpu_bie.NumpyB() if os.environ.get("LOCAL_TEST") else BD.gpu_backend()
    assert B is not None, "GPU path not available: choose a GPU runtime"
    sur = Surrogate(MODEL)
    S = shapes()[:int(os.environ.get("NSHAPE", "24"))]; print(f"{len(S)} test stars, k = {KS}, HNA = {DO_HNA}", flush=True)
    # check 1: the k = 512 reference reproduces a far-field value of the original benchmark (test_id sample 244)
    i, j, re, im = [c for c in BD.CHECKS["test_id"] if c[0] == 244][0]
    m244 = meta("test_id")[244]; cv = BD.shape_of(m244["params"])
    F, _, _ = reference(B, cv, m244["k"], [m244["alpha"]])
    e = abs(F[0, j] - complex(re, im)) / abs(complex(re, im))
    print(f"check: k=512 reference vs original benchmark, rel diff {e:.1e}", flush=True); assert e < 1e-5
    # check 2: resolution of the reference at the largest k (one shape)
    cv = star_curve(S[0]["params"]["a"], S[0]["params"]["b"], S[0]["params"]["r0"])
    F55, N55, t55 = reference(B, cv, max(KS), S[0]["alphas"][:1])
    F70, N70, t70 = reference(B, cv, max(KS), S[0]["alphas"][:1], ppw=7.0)
    print(f"check: k={max(KS)} reference 5.5 ppw (N={N55}, {t55:.0f}s) vs 7 ppw (N={N70}, {t70:.0f}s): rel diff {rel(F55[0], F70[0]):.1e}", flush=True)
    res = dict(ks=KS, model=os.path.basename(MODEL), ref_resolution_check=dict(k=max(KS), rel_diff=rel(F55[0], F70[0])), rows=[])
    for k in KS:
        t0 = time.time()
        for si, sh in enumerate(S):
            p = sh["params"]; cv = star_curve(p["a"], p["b"], p["r0"]); al = np.array(sh["alphas"])
            Fref, N, tref = reference(B, cv, k, al)
            surrogate_farfield(sur, cv, k, al[:1], XH)                                  # warm-up
            ts = time.perf_counter(); Fs, _, se = surrogate_farfield(sur, cv, k, al, XH); ts = (time.perf_counter() - ts) / len(al)
            E = se.farfield_operator(XH)
            for jj, a in enumerate(al):
                ch = se.charts(a); z = np.zeros((32, 8), complex)
                base = dict(k=k, shape=si, alpha=float(a))
                res["rows"] += [dict(base, method="surrogate", err=rel(Fs[:, jj], Fref[jj]), time=ts),
                                dict(base, method="fock", err=rel(E @ se.density(ch, z), Fref[jj])),
                                dict(base, method="po", err=rel(E @ se.density(ch, z, base_kind="po"), Fref[jj])),
                                dict(base, method="nystrom_ref", nodes=N, gpu_time=tref / len(al))]
            if DO_HNA:
                rows, tap = hna(B, cv, k, al[0], Fref[0])
                res["rows"] += [dict(r, k=k, shape=si, alpha=float(al[0]), gpu_apply_time=tap) for r in rows]
            json.dump(res, open(OUT, "w"))
            last = [r for r in res["rows"] if r["k"] == k and r["shape"] == si]
            msg = " ".join(f"{r['method']} {r['err']:.1e}" for r in last if "err" in r and (r["method"] in ("surrogate", "fock") or r["method"].endswith("d16")) and r["alpha"] == float(al[0]))
            print(f"k={k} shape {si + 1}/{len(S)} N={N}  {msg}  ({time.time() - t0:.0f}s)", flush=True)
        # summary for this k
        print(f"== k = {k}: median relative far-field error", flush=True)
        for meth in sorted({r["method"] for r in res["rows"] if r["k"] == k and "err" in r}):
            e = [r["err"] for r in res["rows"] if r["k"] == k and r["method"] == meth]
            dof = [r["dof"] for r in res["rows"] if r["k"] == k and r["method"] == meth and "dof" in r]
            print(f"   {meth:12s} {np.median(e):.2e}  (max {np.max(e):.2e}, n={len(e)}{', dof %d' % np.median(dof) if dof else ''})", flush=True)
    print("saved", OUT, f"({os.path.getsize(OUT) / 1e3:.0f} kB)", flush=True)


if __name__ == "__main__":
    main()
