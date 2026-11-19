"""Theory figure (Section 3): fixed point versus geometric co-moving points along the rotation family.
Data: fig_theory_data.npz (fig_theory_data.py)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.6})
H = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(H, "fig_theory_data.npz")); ts = d["ts"]; XI = d["XI"]; ks = d["ks"]
col = {32: "#9bbb59", 64: "#77a0c9", 128: "#4f81bd", 256: "#2f5f95", 512: "#1f3864"}
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.55), gridspec_kw=dict(width_ratios=[1.15, 0.85, 1.15]))
for k in (32, 128, 512):
    ax[0].plot(ts, abs(d[f"h_{k}"]), color=col[k], lw=1.0, label=f"$k={k}$")
ax[0].plot(ts, abs(d["kink"]), color="k", lw=0.9, ls="--", label=r"$|2i\min\{\varphi_n,0\}|$")
ax[0].set_xlabel("parameter $t$ (rotation, rad)"); ax[0].set_ylabel(r"$|h_k(t)|$")
ax[0].set_title("(a) fixed point crossed by a shadow point", fontsize=8); ax[0].legend(frameon=False, loc="upper right")
ax[0].set_xlim(ts[0], ts[-1])
e = np.array([abs(d[f"h_{k}"] - d["kink"]).max() for k in ks])
ax[1].loglog(ks, e, "o-", color="#1f4e79", ms=3.5, lw=1, label=r"$\max_t|h_k-K|$")
kk = np.array([24, 700]); ax[1].loglog(kk, 0.855 * kk ** (-1 / 3), ":", color="0.4", lw=0.9, label=r"$0.855\,k^{-1/3}$")
ax[1].set_xlabel("$k$"); ax[1].set_title("(b) distance to the kink", fontsize=8); ax[1].legend(frameon=False, loc="lower left")
ax[1].set_xticks([32, 64, 128, 256, 512]); ax[1].set_xticklabels(["32", "64", "128", "256", "512"]); ax[1].minorticks_off()
ax[1].set_ylim(0.08, 0.4); ax[1].set_yticks([0.1, 0.2, 0.3]); ax[1].set_yticklabels(["0.1", "0.2", "0.3"])
ls = {-2.0: "-", 0.0: "--", 1.0: ":"}
for k in (32, 128, 512):
    for i, xi in enumerate(XI):
        ax[2].plot(ts, abs(d[f"w_{k}"][i]), color=col[k], lw=1.0, ls=ls[float(xi)])
for xi, y in zip(XI, (1.75, 1.05, 0.2)):
    ax[2].text(0.0, y, rf"$\xi={xi:g}$", ha="center", fontsize=7, color="0.25")
for k in (32, 128, 512):
    ax[2].plot([], [], color=col[k], lw=1.0, label=f"$k={k}$")
ax[2].set_xlabel("parameter $t$ (rotation, rad)"); ax[2].set_ylabel(r"$|\tilde w_k(t)|/e_k(\sigma)$")
ax[2].set_title(r"(c) co-moving points, fixed $\xi=\lambda\sigma$", fontsize=8); ax[2].legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.13))
ax[2].set_xlim(ts[0], ts[-1]); ax[2].set_ylim(0, 1.95)
fig.tight_layout(w_pad=0.8)
fig.savefig(os.path.join(H, "fig_theory.pdf")); fig.savefig(os.path.join(H, "fig_theory.png"), dpi=220)
print("errors*k^{1/3}:", (e * ks ** (1 / 3)).round(3))
