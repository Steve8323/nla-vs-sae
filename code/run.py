"""
NLA vs SAE — FULL EXPERIMENT.

Do two unsupervised interpretability methods agree on the SAME Qwen2.5-7B
residual-stream token activations?

  NLA (kitft, Anthropic): activation -> English description   (reads hidden_states[21])
  SAE (andyrdt):          activation -> sparse feature codes   (reads hidden_states[20])

Both are autoencoders of the residual stream, so we can compare them WITHOUT any
external feature labels, three ways:

  (A) Feature-conditioned NLA consistency [headline]:
      do tokens that fire the SAME SAE feature get MORE-similar NLA descriptions
      than random token pairs? (lift over baseline; shuffled control must be ~0)
  (B) Similarity-structure agreement:
      Spearman corr between the NLA-text similarity matrix and the SAE-feature
      similarity matrix over all token pairs (shuffled control must be ~0).
  (C) Autoencoder fidelity head-to-head:
      SAE decode cosine vs NLA AR-reconstruction cosine on the same tokens.

Run:  modal run run.py            (defaults: 36 docs x 6 positions)
"""

import modal

# Same validated image as probe.py (already cached). No new deps -> no rebuild.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install("sglang[all]==0.5.6")
    .pip_install(
        "flashinfer-python",
        extra_options="--find-links https://flashinfer.ai/whl/cu124/torch2.5/",
    )
    .pip_install(
        "transformers>=4.48", "safetensors", "httpx", "orjson",
        "pyyaml", "numpy", "pyarrow", "huggingface_hub",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/kitft/natural_language_autoencoders.git /root/nla_repo"
    )
    .env({"HF_HOME": "/cache/hf"})
    .apt_install("libnuma1", "libnuma-dev")
    .pip_install("accelerate")
)

app = modal.App("nla-sae-run", image=image)
hf_cache = modal.Volume.from_name("nla-hf-cache", create_if_missing=True)

QWEN = "Qwen/Qwen2.5-7B-Instruct"
AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"
AR_REPO = "kitft/nla-qwen2.5-7b-L20-ar"
SAE_REPO = "andyrdt/saes-qwen2.5-7b-instruct"
SAE_SUBDIR = "resid_post_layer_19/trainer_0"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HS_IDX_NLA = 21
HS_IDX_SAE = 20
SGLANG_PORT = 30000

# Diverse paragraphs -> varied residual-stream content across many topics.
DOCS = [
    "The mitochondria is the powerhouse of the cell, producing ATP through aerobic respiration. Glucose is broken down in a series of enzyme-catalyzed steps.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen. Chlorophyll in the chloroplasts absorbs light most strongly in the blue and red bands.",
    "In contract law, an offer must be accepted unconditionally for a binding agreement to form. Consideration is the price paid for the promise, and it must have legal value.",
    "The defendant argued that the search violated the Fourth Amendment because no warrant had been issued. The judge weighed the exclusionary rule against the good-faith exception.",
    "To make a proper risotto, toast the arborio rice in butter before adding warm stock one ladle at a time. Stir constantly so the starch releases and the texture turns creamy.",
    "Preheat the oven to 220 degrees and score the bread dough before baking. Steam in the first ten minutes gives the crust its shine and crackle.",
    "The central bank raised interest rates by half a point to combat persistent inflation. Bond yields climbed while equity markets sold off on fears of slower growth.",
    "Diversifying a portfolio across uncorrelated assets reduces variance without necessarily lowering expected return. Rebalancing enforces a disciplined buy-low sell-high policy.",
    "In the final minutes the striker curled a free kick over the wall and into the top corner. The goalkeeper dove but could not reach the swerving ball.",
    "The marathon runner hit the wall at mile twenty as her glycogen stores ran dry. She slowed her pace, took a gel, and pushed through to the finish.",
    "The recursive function computes the factorial by multiplying n with the factorial of n minus one. The base case returns one when the argument reaches zero.",
    "A hash map offers average constant-time lookups by mapping keys to buckets through a hash function. Collisions are resolved with chaining or open addressing.",
    "The Roman Republic gave way to empire after Augustus consolidated power following years of civil war. The Senate retained a facade of authority while real control shifted.",
    "The printing press spread literacy across Europe and accelerated the Reformation. Pamphlets could now reach thousands of readers within weeks rather than years.",
    "The violinist tuned to the oboe's A before the concerto began. The orchestra swelled through the development section as the soloist traded phrases with the winds.",
    "Jazz improvisation builds on the underlying chord changes while bending rhythm and pitch. A good soloist listens as much as they play, answering the rhythm section.",
    "A cold front swept through overnight, dropping temperatures and bringing gusty winds. By morning the sky had cleared to a brilliant, dry blue.",
    "Hurricanes draw their energy from warm ocean water and weaken rapidly over land. The eye is a deceptive calm surrounded by the most violent winds of the storm.",
    "She felt a wave of relief wash over her when the test results finally came back clear. The weeks of waiting had left her exhausted and on edge.",
    "Grief comes in waves, he wrote, sometimes gentle and sometimes crushing without warning. There is no schedule to it and no way to hurry it along.",
    "Quantum entanglement links two particles so that measuring one instantly constrains the other. This correlation holds regardless of the distance separating them.",
    "General relativity describes gravity as the curvature of spacetime caused by mass and energy. Light bends as it passes a massive star, an effect confirmed during an eclipse.",
    "The patient presented with a fever, a productive cough, and shortness of breath. The chest X-ray showed consolidation consistent with bacterial pneumonia.",
    "Vaccines train the immune system by presenting a harmless piece of a pathogen. Memory B cells then respond faster if the real infection ever arrives.",
    "The Amazon basin holds the largest rainforest on Earth and a vast share of its biodiversity. Its rivers carry more water than the next several largest combined.",
    "The Himalayas rose as the Indian plate collided with Asia and continue to grow each year. Glaciers there feed rivers that sustain billions of people downstream.",
    "In Norse myth, Odin sacrificed an eye at Mimir's well to gain wisdom. He hung from the world tree for nine nights to learn the secrets of the runes.",
    "The hero of the epic must journey to the underworld and return changed. This descent and return is a pattern found in myths across many cultures.",
    "The neural network adjusts its weights by backpropagating the gradient of the loss. Learning rate and batch size strongly shape how training converges.",
    "Attention lets a transformer weigh every token against every other in the sequence. This is expensive but captures long-range dependencies that recurrence struggles with.",
    "The vineyard harvests at dawn to keep the grapes cool and preserve acidity. Fermentation temperature decides whether the wine turns crisp or round.",
    "Espresso extraction depends on grind size, dose, pressure, and water temperature. A shot that runs too fast tastes sour and thin, too slow tastes bitter.",
    "The satellite entered a low orbit and adjusted its attitude with small thruster bursts. Ground control uploaded the imaging schedule for the next pass over the pole.",
    "Tectonic plates grind past one another along the fault, storing elastic strain for decades. When it releases, the ground lurches and the shaking radiates outward.",
    "The chess grandmaster sacrificed a knight to expose the enemy king. Three moves later the attack broke through and resignation followed.",
    "Bees communicate the direction and distance of flowers through a waggle dance. The angle of the dance encodes the bearing relative to the sun.",
]


@app.function(gpu="H100", volumes={"/cache": hf_cache}, timeout=3600)
def run(n_docs: int = 36, pos_per_doc: int = 6, skip_first: int = 12,
        do_critic: bool = True):
    import os, sys, json, time, subprocess, gc
    import numpy as np
    import torch

    sys.path.insert(0, "/root/nla_repo")
    os.environ.setdefault("HF_HOME", "/cache/hf")
    from huggingface_hub import hf_hub_download, snapshot_download

    def banner(t):
        print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)

    docs = DOCS[:n_docs]

    # ---------- PHASE A: extract activations + SAE encode/decode --------------
    banner(f"PHASE A — extract Qwen activations ({len(docs)} docs) + SAE")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN)
    model = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.bfloat16).to("cuda").eval()

    X20, X21, records = [], [], []
    for di, doc in enumerate(docs):
        ids = tok(doc, return_tensors="pt").input_ids.to("cuda")
        T = ids.shape[1]
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        cand = list(range(skip_first, T))
        if not cand:
            continue
        # evenly spaced positions across the usable range
        step = max(1, len(cand) // pos_per_doc)
        chosen = cand[::step][:pos_per_doc]
        for p in chosen:
            h20 = hs[HS_IDX_SAE][0, p].float().cpu().numpy()
            h21 = hs[HS_IDX_NLA][0, p].float().cpu().numpy()
            n21 = float(np.linalg.norm(h21))
            if n21 > 300 or n21 < 30:  # skip rare OOD-norm positions (decode poorly)
                continue
            X20.append(h20); X21.append(h21)
            records.append({
                "doc": di,
                "pos": int(p),
                "token": tok.decode(ids[0, p]),
                "context": tok.decode(ids[0, max(0, p - 8):p + 1]),
            })
    X20 = np.stack(X20); X21 = np.stack(X21)
    N = X20.shape[0]
    print(f"collected {N} token activations")

    # SAE: encode (threshold BatchTopK) + decode, on GPU
    sae_cfg = json.load(open(hf_hub_download(SAE_REPO, f"{SAE_SUBDIR}/config.json")))
    sd = torch.load(hf_hub_download(SAE_REPO, f"{SAE_SUBDIR}/ae.pt"), map_location="cuda")
    W_enc = sd["encoder.weight"].float()      # [dict, d]
    b_enc = sd["encoder.bias"].float()        # [dict]
    W_dec = sd["decoder.weight"].float()      # [d, dict]
    b_dec = sd["b_dec"].float()               # [d]
    thr = float(sd["threshold"])
    Xt = torch.tensor(X20, device="cuda")
    pre = torch.relu((Xt - b_dec) @ W_enc.t() + b_enc)
    feats = torch.where(pre > thr, pre, torch.zeros_like(pre))     # [N, dict]
    recon = feats @ W_dec.t() + b_dec
    sae_cos = torch.nn.functional.cosine_similarity(Xt, recon, dim=-1).cpu().numpy()
    # normalized feature-similarity matrix (cosine over sparse codes)
    fn = torch.nn.functional.normalize(feats, dim=-1)
    S_sae = (fn @ fn.t()).cpu().numpy()
    # per-token active features (indices + values), keep top-20 for the report
    active_sets = []
    for i in range(N):
        nz = torch.nonzero(feats[i] > 0, as_tuple=False).flatten()
        vals = feats[i, nz]
        order = torch.argsort(vals, descending=True)
        idx_sorted = nz[order].cpu().tolist()
        val_sorted = vals[order].cpu().tolist()
        active_sets.append(idx_sorted)
        records[i]["sae_recon_cos"] = float(sae_cos[i])
        records[i]["n_active"] = len(idx_sorted)
        records[i]["top_features"] = [
            {"feat": int(f), "val": round(float(v), 3)}
            for f, v in list(zip(idx_sorted, val_sorted))[:20]
        ]
    n_active_arr = np.array([r["n_active"] for r in records])
    print(f"SAE: mean active feats/token = {n_active_arr.mean():.1f}, "
          f"median recon cos = {np.median(sae_cos):.3f}")

    del model, W_enc, b_enc, W_dec, b_dec, Xt, pre, feats, recon, fn
    gc.collect(); torch.cuda.empty_cache()

    # ---------- PHASE B: NLA verbalize (SGLang AV) ----------------------------
    banner("PHASE B — NLA verbalize via SGLang AV")
    av_dir = snapshot_download(AV_REPO)
    server = subprocess.Popen(
        [sys.executable, "-m", "sglang.launch_server",
         "--model-path", av_dir, "--port", str(SGLANG_PORT),
         "--disable-radix-cache", "--mem-fraction-static", "0.6",
         "--trust-remote-code"],
        env=dict(os.environ),
    )
    import httpx
    url = f"http://localhost:{SGLANG_PORT}"
    ready = False
    for i in range(150):
        if server.poll() is not None:
            raise RuntimeError(f"sglang exited early rc={server.returncode}")
        try:
            if httpx.get(f"{url}/get_model_info", timeout=2).status_code == 200:
                ready = True; break
        except Exception:
            pass
        time.sleep(5)
    print(f"sglang ready={ready} after ~{(i+1)*5}s")

    import nla_inference
    client = nla_inference.NLAClient(av_dir, sglang_url=url, device="cpu")
    t0 = time.time()
    texts = client.generate_batch(list(X21), temperature=0.0, max_new_tokens=220)
    print(f"verbalized {len(texts)} activations in {time.time()-t0:.0f}s")
    for i, t in enumerate(texts):
        records[i]["nla_text"] = t

    server.terminate()
    try:
        server.wait(timeout=30)
    except Exception:
        server.kill()
    time.sleep(3); gc.collect(); torch.cuda.empty_cache()

    # ---------- PHASE C: NLA AR reconstruction fidelity (optional) ------------
    nla_cos = None
    if do_critic:
        banner("PHASE C — NLA AR reconstruction (fidelity head-to-head)")
        try:
            ar_dir = snapshot_download(AR_REPO)
            critic = nla_inference.NLACritic(ar_dir, device="cuda:0")
            cosses = []
            for i in range(N):
                _, cos = critic.score(records[i]["nla_text"], X21[i])
                cosses.append(float(cos))
                records[i]["nla_recon_cos"] = float(cos)
            nla_cos = np.array(cosses)
            print(f"NLA AR: median recon cos = {np.median(nla_cos):.3f}")
            del critic; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"critic skipped: {e}")

    # ---------- PHASE D: text embeddings + agreement metrics ------------------
    banner("PHASE D — embed NLA texts (MiniLM) + agreement metrics")
    from transformers import AutoModel
    emb_tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    emb_model = AutoModel.from_pretrained(EMBED_MODEL).to("cuda").eval()

    def embed(text_list):
        out = []
        for j in range(0, len(text_list), 32):
            batch = text_list[j:j + 32]
            enc = emb_tok(batch, padding=True, truncation=True, max_length=256,
                          return_tensors="pt").to("cuda")
            with torch.no_grad():
                m = emb_model(**enc).last_hidden_state
            mask = enc.attention_mask.unsqueeze(-1).float()
            pooled = (m * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
        return torch.cat(out).numpy()

    E = embed([r["nla_text"] for r in records])
    S_text = E @ E.T

    iu = np.triu_indices(N, k=1)
    text_pairs = S_text[iu]
    sae_pairs = S_sae[iu]
    baseline_sim = float(text_pairs.mean())

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        ra = (ra - ra.mean()) / (ra.std() + 1e-12)
        rb = (rb - rb.mean()) / (rb.std() + 1e-12)
        return float((ra * rb).mean())

    # (A) feature-conditioned NLA consistency
    support = {}
    for i, feats_i in enumerate(active_sets):
        for f in feats_i:
            support.setdefault(f, []).append(i)
    qualifying = {f: ts for f, ts in support.items() if len(ts) >= 3}

    def within_sim(ts):
        if len(ts) < 2:
            return None
        s = [S_text[a, b] for k, a in enumerate(ts) for b in ts[k + 1:]]
        return float(np.mean(s))

    lifts = []
    for f, ts in qualifying.items():
        w = within_sim(ts)
        if w is not None:
            lifts.append(w - baseline_sim)
    lifts = np.array(lifts)
    # shuffled control: permute the text embeddings across tokens
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    S_text_shuf = S_text[np.ix_(perm, perm)]
    shuf_lifts = []
    for f, ts in qualifying.items():
        if len(ts) >= 2:
            s = [S_text_shuf[a, b] for k, a in enumerate(ts) for b in ts[k + 1:]]
            shuf_lifts.append(float(np.mean(s)) - baseline_sim)
    shuf_lifts = np.array(shuf_lifts)

    # (B) structure agreement + shuffled control
    rho = spearman(text_pairs, sae_pairs)
    rho_shuf = spearman(text_pairs, S_sae[np.ix_(perm, perm)][iu])

    metrics = {
        "n_tokens": int(N),
        "sae": {
            "mean_active_features": float(n_active_arr.mean()),
            "median_recon_cos": float(np.median(sae_cos)),
            "dict_size": sae_cfg["trainer"]["dict_size"],
            "layer": sae_cfg["trainer"]["layer"],
        },
        "nla": {
            "median_recon_cos": (None if nla_cos is None else float(np.median(nla_cos))),
            "layer_hs_index": HS_IDX_NLA,
        },
        "agreement_A_feature_conditioned": {
            "n_features_ge3": int(len(qualifying)),
            "mean_lift": float(lifts.mean()) if len(lifts) else None,
            "frac_positive": float((lifts > 0).mean()) if len(lifts) else None,
            "shuffled_control_mean_lift": float(shuf_lifts.mean()) if len(shuf_lifts) else None,
            "baseline_pair_sim": baseline_sim,
        },
        "agreement_B_structure": {
            "spearman_text_vs_sae": rho,
            "shuffled_control_spearman": rho_shuf,
        },
    }
    print(json.dumps(metrics, indent=2))

    # qualitative: top features by support -> tokens + NLA texts
    top_feats = sorted(qualifying.items(), key=lambda kv: -len(kv[1]))[:12]
    qualitative = []
    for f, ts in top_feats:
        qualitative.append({
            "feature": int(f),
            "n_tokens": len(ts),
            "members": [{
                "token": records[i]["token"],
                "context": records[i]["context"],
                "nla_text": records[i]["nla_text"],
            } for i in ts[:6]],
        })

    hf_cache.commit()
    return {"metrics": metrics, "records": records, "qualitative": qualitative,
            "config": {"n_docs": len(docs), "pos_per_doc": pos_per_doc}}


@app.local_entrypoint()
def main(n_docs: int = 36, pos_per_doc: int = 6):
    import json, datetime, pathlib, html

    res = run.remote(n_docs=n_docs, pos_per_doc=pos_per_doc)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path.home() / "nla_sae_compare"
    (out / f"results_{ts}.json").write_text(json.dumps(res, indent=2))

    m = res["metrics"]
    A = m["agreement_A_feature_conditioned"]
    B = m["agreement_B_structure"]

    def esc(s): return html.escape(str(s))
    rows = ""
    for q in res["qualitative"]:
        mem = "".join(
            f"<tr><td><code>{esc(repr(x['token']))}</code></td>"
            f"<td class=ctx>{esc(x['context'])}</td>"
            f"<td class=nla>{esc(x['nla_text'][:400])}</td></tr>"
            for x in q["members"]
        )
        rows += (f"<h3>SAE feature #{q['feature']} — fires on {q['n_tokens']} tokens</h3>"
                 f"<table class=grp><tr><th>token</th><th>context</th>"
                 f"<th>NLA description</th></tr>{mem}</table>")

    report = f"""<!doctype html><meta charset=utf-8>
<title>NLA vs SAE agreement — {ts}</title>
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
 h3{{margin-top:1.6rem}}
</style>
<h1>NLA vs SAE — do they agree on the same activations?</h1>
<p class=sub>Qwen2.5-7B-Instruct · NLA hidden_states[21] (kitft) vs SAE resid_post_layer_19 (andyrdt) ·
{m['n_tokens']} token activations · {ts}</p>

<div class=cards>
 <div class=card><div>A · Feature-conditioned NLA consistency</div>
   <div class="big good">+{A['mean_lift']:.3f}</div>
   <div class=muted>mean NLA-similarity lift for tokens sharing an SAE feature<br>
   {A['frac_positive']*100:.0f}% of {A['n_features_ge3']} features positive ·
   shuffled control {A['shuffled_control_mean_lift']:+.3f}</div></div>
 <div class=card><div>B · Similarity-structure agreement</div>
   <div class="big good">ρ={B['spearman_text_vs_sae']:.3f}</div>
   <div class=muted>Spearman: NLA-text sim vs SAE-feature sim over all pairs<br>
   shuffled control ρ={B['shuffled_control_spearman']:.3f}</div></div>
 <div class=card><div>C · Autoencoder fidelity</div>
   <div class=big>{m['sae']['median_recon_cos']:.2f} <span class=muted>/</span> {(m['nla']['median_recon_cos'] or 0):.2f}</div>
   <div class=muted>median recon cosine — SAE decode / NLA AR<br>
   SAE mean active feats/token: {m['sae']['mean_active_features']:.0f}</div></div>
</div>

<p><b>Read:</b> if A's lift is clearly positive (and its shuffled control ~0) and B's ρ is
positive (control ~0), the two methods agree: tokens the SAE groups together are described
similarly by the NLA, and their pairwise-similarity structures track each other.</p>

<h2>Qualitative — top SAE features and how the NLA describes their tokens</h2>
{rows}
"""
    rpath = out / f"report_{ts}.html"
    rpath.write_text(report)
    print("\nRESULTS:", out / f"results_{ts}.json")
    print("REPORT :", rpath)
