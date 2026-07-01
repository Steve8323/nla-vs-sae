"""Generate figures for the paper directly from results.json (real data only)."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
res = json.load(open(ROOT / "results" / "results.json"))
m = res["metrics"]
recs = res["records"]
A = m["agreement_A_feature_conditioned"]
B = m["agreement_B_structure"]

base = A["baseline_pair_sim"]
within = base + A["mean_lift"]
shuf = base + A["shuffled_control_mean_lift"]

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ---- Figure 1: the two agreement tests -------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))

ax[0].bar(["random\npairs", "same SAE\nfeature", "shuffled\ncontrol"],
          [base, within, shuf],
          color=["#9aa0a6", "#137333", "#c8c8c8"])
ax[0].set_ylabel("mean NLA-description cosine")
ax[0].set_title(f"A · feature-conditioned consistency\n(+{A['mean_lift']:.3f} lift, "
                f"{A['frac_positive']*100:.0f}% of {A['n_features_ge3']} features)")
for i, v in enumerate([base, within, shuf]):
    ax[0].text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
ax[0].set_ylim(0, max(within, base) * 1.25)

ax[1].bar(["NLA vs SAE", "shuffled\ncontrol"],
          [B["spearman_text_vs_sae"], B["shuffled_control_spearman"]],
          color=["#1a73e8", "#c8c8c8"])
ax[1].axhline(0, color="#444", lw=0.8)
ax[1].set_ylabel("Spearman ρ  (all token pairs)")
ax[1].set_title("B · similarity-structure agreement")
for i, v in enumerate([B["spearman_text_vs_sae"], B["shuffled_control_spearman"]]):
    ax[1].text(i, v + (0.006 if v >= 0 else -0.018), f"{v:+.3f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig_agreement.png", dpi=150)
print("wrote figures/fig_agreement.png")

# ---- Figure 2: autoencoder fidelity head-to-head ---------------------------
sae = np.array([r["sae_recon_cos"] for r in recs])
nla = np.array([r.get("nla_recon_cos") for r in recs if r.get("nla_recon_cos") is not None])
fig2, ax2 = plt.subplots(figsize=(6.4, 3.8))
bins = np.linspace(min(sae.min(), nla.min()) - 0.02, 1.0, 30)
ax2.hist(sae, bins=bins, alpha=0.55, label=f"SAE decode (median {np.median(sae):.3f})", color="#137333")
ax2.hist(nla, bins=bins, alpha=0.55, label=f"NLA AR (median {np.median(nla):.3f})", color="#1a73e8")
ax2.axvline(np.median(sae), color="#137333", ls="--", lw=1)
ax2.axvline(np.median(nla), color="#1a73e8", ls="--", lw=1)
ax2.set_xlabel("reconstruction cosine with original activation")
ax2.set_ylabel("token count")
ax2.set_title("C · both are faithful residual-stream autoencoders")
ax2.legend()
fig2.tight_layout()
fig2.savefig(ROOT / "figures" / "fig_fidelity.png", dpi=150)
print("wrote figures/fig_fidelity.png")
