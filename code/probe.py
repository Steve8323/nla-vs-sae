"""
NLA vs SAE — PROBE (cheap de-risk run before the full experiment).

Goal of the full experiment: on Qwen2.5-7B-Instruct residual-stream activations,
run the open NLA (Anthropic / kitft) and an open SAE (andyrdt) on the SAME token
activations and measure whether their interpretations agree.

This PROBE validates the three risky integration points on ~2 activations only,
so the first real GPU spend can't be wasted on a broken pipeline:
  1. SAE loads from ae.pt and encodes/decodes a real Qwen activation.
  2. Qwen2.5-7B extraction: hidden_states[20] (SAE) and hidden_states[21] (NLA).
  3. SGLang serves the AV checkpoint via input_embeds and verbalizes activations.

Run:  modal run probe.py
"""

import modal

# ---- image: sglang + transformers + the NLA inference client -----------------
# sglang[all] pulls a CUDA torch; flashinfer from its own index. Everything else
# is layered on top without forcing torch.
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
        "transformers>=4.48",
        "safetensors",
        "httpx",
        "orjson",
        "pyyaml",
        "numpy",
        "pyarrow",
        "huggingface_hub",
    )
    # nla_inference.py (single-file AV client + NLACritic) lives in the repo.
    .run_commands(
        "git clone --depth 1 https://github.com/kitft/natural_language_autoencoders.git /root/nla_repo"
    )
    .env({"HF_HOME": "/cache/hf"})
    # --- probe-2 fixes (appended so sglang/flashinfer layers stay cached) ---
    # libnuma: sgl_kernel .so needs libnuma.so.1 at load time.
    # accelerate: transformers device_map/.to(cuda) path wants it.
    .apt_install("libnuma1", "libnuma-dev")
    .pip_install("accelerate")
)

app = modal.App("nla-sae-probe", image=image)
hf_cache = modal.Volume.from_name("nla-hf-cache", create_if_missing=True)

# ---- constants ---------------------------------------------------------------
QWEN = "Qwen/Qwen2.5-7B-Instruct"
AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"
SAE_REPO = "andyrdt/saes-qwen2.5-7b-instruct"
SAE_SUBDIR = "resid_post_layer_19/trainer_0"  # k=32, dict=131072, d=3584
# NLA reads "layer 20" = hidden_states[21] (output of block 20).
# SAE resid_post_layer_19 = hidden_states[20] (output of block 19). 1 block apart.
HS_IDX_NLA = 21
HS_IDX_SAE = 20
SGLANG_PORT = 30000


@app.function(gpu="H100", volumes={"/cache": hf_cache}, timeout=2400)
def probe():
    import os, sys, json, time, subprocess, traceback
    import numpy as np
    import torch

    sys.path.insert(0, "/root/nla_repo")
    os.environ.setdefault("HF_HOME", "/cache/hf")
    from huggingface_hub import hf_hub_download, snapshot_download

    def banner(t):
        print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)

    report = {}

    # ---------- STAGE 1: SAE format + encode/decode ---------------------------
    banner("STAGE 1 — SAE (andyrdt) format, load, encode/decode")
    try:
        cfg_path = hf_hub_download(SAE_REPO, f"{SAE_SUBDIR}/config.json")
        ae_path = hf_hub_download(SAE_REPO, f"{SAE_SUBDIR}/ae.pt")
        sae_cfg = json.load(open(cfg_path))
        print("SAE config trainer:", json.dumps(sae_cfg.get("trainer", {}), indent=2))
        sd = torch.load(ae_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and not any(
            hasattr(v, "shape") for v in sd.values()
        ):
            sd = sd["state_dict"]
        print("\nae.pt top-level type:", type(sd).__name__)
        print("ae.pt keys -> shapes / values:")
        for k, v in (sd.items() if isinstance(sd, dict) else []):
            try:
                print(f"   {k:30s} {tuple(v.shape)} {v.dtype}")
            except Exception:
                print(f"   {k:30s} {repr(v)[:60]}")
        report["sae_keys"] = list(sd.keys()) if isinstance(sd, dict) else str(type(sd))

        # Best-effort generic BatchTopK encode/decode from whatever keys exist.
        def pick(*names):
            for n in names:
                if n in sd:
                    return sd[n]
            return None

        W_enc = pick("encoder.weight", "W_enc", "encoder.W_enc")
        b_enc = pick("encoder.bias", "b_enc", "encoder.b_enc")
        W_dec = pick("decoder.weight", "W_dec", "decoder.W_dec")
        b_dec = pick("b_dec", "decoder.bias", "bias")
        thr = pick("threshold", "k_threshold")
        k = sae_cfg.get("trainer", {}).get("k", 32)
        d = sae_cfg.get("trainer", {}).get("activation_dim", 3584)
        print(f"\nresolved: W_enc={None if W_enc is None else tuple(W_enc.shape)} "
              f"b_enc={None if b_enc is None else tuple(b_enc.shape)} "
              f"W_dec={None if W_dec is None else tuple(W_dec.shape)} "
              f"b_dec={None if b_dec is None else tuple(b_dec.shape)} "
              f"thr={None if thr is None else (float(thr) if thr.numel()==1 else tuple(thr.shape))} k={k}")

        # Confirmed BatchTopKSAE layout:
        #   encoder.weight [dict,d], encoder.bias [dict], decoder.weight [d,dict],
        #   b_dec [d], threshold scalar. Inference uses the threshold (not top-k).
        x = torch.randn(2, d) * (130.0 / (d ** 0.5))  # ~typical Qwen L20 norm
        pre = torch.relu((x - b_dec) @ W_enc.t() + b_enc)
        feats = torch.where(pre > float(thr), pre, torch.zeros_like(pre))
        recon = feats @ W_dec.t() + b_dec          # [2,dict]@[dict,d] -> [2,d]
        nnz = (feats > 0).sum(-1).tolist()
        cos = torch.nn.functional.cosine_similarity(x, recon, dim=-1).mean().item()
        print(f"manual encode/decode OK: active feats/row={nnz}, recon cos={cos:.3f} "
              f"(random input -> low cos expected; real activations should be high)")
        report["sae_manual_nnz"] = nnz
        report["sae_manual_recon_cos"] = cos
        report["stage1"] = "ok"
    except Exception:
        report["stage1"] = "FAIL"
        traceback.print_exc()

    # ---------- STAGE 2: Qwen extraction (hidden_states[20], [21]) ------------
    banner("STAGE 2 — Qwen2.5-7B extraction of hidden_states[20] & [21]")
    acts_nla, acts_sae = None, None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(QWEN)
        model = AutoModelForCausalLM.from_pretrained(
            QWEN, dtype=torch.bfloat16
        ).to("cuda").eval()
        text = "The mitochondria is the powerhouse of the cell, producing ATP through respiration."
        ids = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        hs = out.hidden_states
        print(f"num hidden_states = {len(hs)} (expect n_layers+1 = 29)")
        # pick two token positions past the first 12 (early positions decode poorly)
        T = hs[0].shape[1]
        pos = [min(T - 1, 14), min(T - 1, 16)]
        pos = sorted(set(p for p in pos if p < T))
        h20 = torch.stack([hs[HS_IDX_SAE][0, p] for p in pos]).float().cpu()
        h21 = torch.stack([hs[HS_IDX_NLA][0, p] for p in pos]).float().cpu()
        print(f"positions {pos}: |h20| norms = {h20.norm(dim=-1).tolist()}, "
              f"|h21| norms = {h21.norm(dim=-1).tolist()}  (Qwen L20 typical ~100-170)")
        acts_nla = h21.numpy()
        acts_sae = h20.numpy()
        report["hidden_states_count"] = len(hs)
        report["h21_norms"] = h21.norm(dim=-1).tolist()
        del model
        torch.cuda.empty_cache()
        report["stage2"] = "ok"
    except Exception:
        report["stage2"] = "FAIL"
        traceback.print_exc()

    # ---------- STAGE 3: SGLang serves AV, verbalize activations --------------
    banner("STAGE 3 — SGLang AV server + verbalize (the input_embeds path)")
    server = None
    try:
        av_dir = snapshot_download(AV_REPO)
        print("AV checkpoint files:", sorted(os.listdir(av_dir)))
        has_meta = any(f.startswith("nla_meta") for f in os.listdir(av_dir))
        print("nla_meta.yaml present:", has_meta)
        report["av_has_meta"] = has_meta

        env = dict(os.environ)
        server = subprocess.Popen(
            [
                sys.executable, "-m", "sglang.launch_server",
                "--model-path", av_dir,
                "--port", str(SGLANG_PORT),
                "--disable-radix-cache",
                "--mem-fraction-static", "0.6",
                "--trust-remote-code",
            ],
            env=env,
        )
        import httpx

        url = f"http://localhost:{SGLANG_PORT}"
        ready = False
        for i in range(120):  # up to ~10 min for weight load
            if server.poll() is not None:
                raise RuntimeError(f"sglang exited early rc={server.returncode}")
            try:
                if httpx.get(f"{url}/get_model_info", timeout=2).status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(5)
        print(f"sglang ready={ready} after ~{(i+1)*5}s")
        report["sglang_ready"] = ready

        if ready and acts_nla is not None:
            import nla_inference

            client = nla_inference.NLAClient(av_dir, sglang_url=url, device="cpu")
            print("NLA cfg d_model:", client.cfg.d_model,
                  "injection_scale:", getattr(client.cfg, "injection_scale", "?"))
            texts = client.generate_batch(
                list(acts_nla), temperature=0.0, max_new_tokens=200
            )
            for i, t in enumerate(texts):
                print(f"\n--- activation {i} verbalization ---\n{t}")
            report["nla_texts"] = texts
        report["stage3"] = "ok"
    except Exception:
        report["stage3"] = "FAIL"
        traceback.print_exc()
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except Exception:
                server.kill()

    hf_cache.commit()
    banner("PROBE SUMMARY")
    print(json.dumps({k: v for k, v in report.items() if k != "nla_texts"}, indent=2))
    return report


@app.local_entrypoint()
def main():
    r = probe.remote()
    print("\n\n==== stages:",
          {k: v for k, v in r.items() if k.startswith("stage")})
