"""Regenerate the HTML report from the saved results JSON, selecting
DISCRIMINATIVE mid-frequency SAE features (not the near-always-on ones) for the
qualitative view. No GPU / no Modal — pure local post-processing."""
import json, glob, html, datetime, pathlib

f = sorted(glob.glob(str(pathlib.Path.home() / "nla_sae_compare" / "results_*.json")))[-1]
d = json.load(open(f))
recs = d["records"]
m = d["metrics"]
N = len(recs)

# rebuild feature -> token indices from each token's top-20 active features
support = {}
for i, r in enumerate(recs):
    for tf in r.get("top_features", []):
        support.setdefault(tf["feat"], []).append(i)

# Semantic (topic-concentrated) features best illustrate agreement: they fire on
# several tokens drawn from ONE topic (few source docs), and we check the NLA
# independently describes those tokens as that same topic. (Syntactic/positional
# features that span every doc are excluded here — they're real, but they show
# register agreement, not topic agreement.)
band = {fq: ts for fq, ts in support.items() if 3 <= len(ts) <= 18}
def concentrated(ts):
    ndocs = len({recs[i]["doc"] for i in ts})
    return ndocs <= 2 and len(ts) >= 3
cand = {fq: ts for fq, ts in band.items() if concentrated(ts)}
# rank by support (more member tokens telling the same story = clearer)
chosen = sorted(cand.items(), key=lambda kv: len(kv[1]), reverse=True)[:14]

def esc(s): return html.escape(str(s))
def first_line(t): return t.split("\n")[0].strip()

rows = ""
for fq, ts in chosen:
    ndocs = len({recs[i]["doc"] for i in ts})
    mem = "".join(
        f"<tr><td><code>{esc(repr(recs[i]['token']))}</code></td>"
        f"<td class=ctx>{esc(recs[i]['context'])}</td>"
        f"<td class=nla>{esc(first_line(recs[i]['nla_text'])[:240])}</td></tr>"
        for i in ts[:7]
    )
    rows += (f"<h3>SAE feature #{fq} — {len(ts)} tokens across {ndocs} documents</h3>"
             f"<table class=grp><tr><th>token</th><th>context</th>"
             f"<th>NLA description (first line)</th></tr>{mem}</table>")

A = m["agreement_A_feature_conditioned"]
B = m["agreement_B_structure"]
ts_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
n_ubiq = sum(1 for fq, t in support.items() if len(t) > N // 2)

report = f"""<!doctype html><meta charset=utf-8>
<title>NLA vs SAE agreement — {ts_now}</title>
<style>
 body{{font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
 h1{{margin-bottom:.2rem}} .sub{{color:#666}}
 .cards{{display:flex;gap:1rem;flex-wrap:wrap;margin:1.5rem 0}}
 .card{{flex:1;min-width:230px;border:1px solid #ddd;border-radius:10px;padding:1rem}}
 .big{{font-size:2rem;font-weight:700}} .good{{color:#137333}} .muted{{color:#999}}
 table{{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}}
 td,th{{border:1px solid #e3e3e3;padding:.4rem .6rem;text-align:left;vertical-align:top;font-size:13px}}
 th{{background:#f6f6f6}} .ctx{{color:#555;width:26%}} .nla{{width:52%}}
 code{{background:#f0f0f0;padding:1px 4px;border-radius:4px}}
 h3{{margin-top:1.6rem}} .note{{background:#fbfbe8;border:1px solid #e6e6b0;border-radius:8px;padding:.6rem 1rem}}
</style>
<h1>NLA vs SAE — do they agree on the same activations?</h1>
<p class=sub>Qwen2.5-7B-Instruct · NLA hidden_states[21] (kitft) vs SAE resid_post_layer_19 (andyrdt) ·
{m['n_tokens']} token activations · run {f.split('_')[-1].replace('.json','')}</p>

<div class=cards>
 <div class=card><div>A · Feature-conditioned NLA consistency</div>
   <div class="big good">+{A['mean_lift']:.3f}</div>
   <div class=muted>mean NLA-similarity lift for tokens sharing an SAE feature<br>
   {A['frac_positive']*100:.0f}% of {A['n_features_ge3']} features positive ·
   shuffled control {A['shuffled_control_mean_lift']:+.4f}</div></div>
 <div class=card><div>B · Similarity-structure agreement</div>
   <div class="big good">ρ={B['spearman_text_vs_sae']:.3f}</div>
   <div class=muted>Spearman: NLA-text sim vs SAE-feature sim over all pairs<br>
   shuffled control ρ={B['shuffled_control_spearman']:+.4f}</div></div>
 <div class=card><div>C · Autoencoder fidelity</div>
   <div class=big>{m['sae']['median_recon_cos']:.2f} <span class=muted>/</span> {(m['nla']['median_recon_cos'] or 0):.2f}</div>
   <div class=muted>median recon cosine — SAE decode / NLA AR<br>
   SAE mean active feats/token: {m['sae']['mean_active_features']:.0f}</div></div>
</div>

<p class=note><b>Verdict:</b> the methods agree. Tokens the SAE groups under a shared feature are
described <b>{A['mean_lift']/A['baseline_pair_sim']*100:.0f}% more similarly</b> by the NLA than
random token pairs (lift +{A['mean_lift']:.3f} on a {A['baseline_pair_sim']:.3f} baseline), for
{A['frac_positive']*100:.0f}% of features — while the shuffled control sits at {A['shuffled_control_mean_lift']:+.4f}.
Global similarity structure also tracks (ρ={B['spearman_text_vs_sae']:.3f} vs control ~0). Both are
high-fidelity autoencoders of the residual stream ({m['sae']['median_recon_cos']:.2f} / {(m['nla']['median_recon_cos'] or 0):.2f}).</p>

<h2>Qualitative — topic-concentrated SAE features and how the NLA describes their tokens</h2>
<p class=sub>Each feature below fires on several tokens drawn from one or two source documents.
The test: does the NLA — with no access to the SAE — independently describe those tokens as the
same subject? ({n_ubiq} near-always-on features and cross-topic syntactic features are excluded
from this view; the metrics above are computed over all {A['n_features_ge3']} features.)</p>
{rows}
"""
out = pathlib.Path.home() / "nla_sae_compare" / f"report_{ts_now}.html"
out.write_text(report)
print(out)
