import numpy as np, json
out={}
for fam,T in [("rotate","0.4"),("mode2","0.08")]:
    for name in ["raw_farfield","raw_density_fixed_angle","envelope_fixed_angle","envelope_fixed_angle_lit_only","envelope_fixed_angle_shadow_pts","comoving_chart_coeffs"]:
        row=[]
        for k in [32,64,128,256,512]:
            e=np.load(f"results/m1_coeffs_{fam}_T{T}_k{k}.npz")[name]; e=e/e[0] if e[0]>0 else e/e.max()
            e=e/e.max()
            # fit range: degrees where envelope (running max from the right) is between 1e-2 and 1e-7
            env=np.maximum.accumulate(e[::-1])[::-1]
            idx=np.where((env<1e-2)&(env>1e-7))[0]
            if len(idx)<3: row.append(None); continue
            s=np.polyfit(idx,np.log(env[idx]),1)[0]; row.append(float(np.exp(-s)))
        out[f"{fam}/{name}"]=row
        print(f"{fam:6s} {name:32s}", " ".join("  -  " if r is None else f"{r:6.3f}" for r in row))
json.dump(out,open("results/m1_rho_fit.json","w"),indent=1)
