# NLA vs SAE — do two interpretability methods agree on the same activations?

A sparse autoencoder (**SAE**) explains a language-model activation as a sparse
code of learned features. A natural-language autoencoder (**NLA**) explains the
*same* activation as a short English description. They are trained
independently, with different objectives, into different output spaces. So: **on
the same activations, do they recover the same structure?**

Run on `Qwen2.5-7B-Instruct` (open NLA from [kitft](https://github.com/kitft/natural_language_autoencoders),
open SAE from [`andyrdt/saes-qwen2.5-7b-instruct`](https://huggingface.co/andyrdt/saes-qwen2.5-7b-instruct)),
over 216 residual-stream token activations — **yes, they agree.**

![agreement](figures/fig_agreement.png)

| Test | Result | Shuffled control |
|---|---|---|
| **A ·** tokens sharing an SAE feature get similar NLA descriptions | lift **+0.189** (≈+79% over baseline); **88%** of 522 features positive | +0.0006 |
| **B ·** NLA-similarity structure tracks SAE-similarity structure | Spearman **ρ = 0.213** (23,220 pairs) | −0.018 |
| **C ·** both are faithful residual-stream autoencoders | SAE decode **0.90** / NLA AR **0.92** median cosine | — |

Topic-concentrated SAE features get matching NLA topic descriptions — biology,
Amazon rainforest, geology, winemaking, contract law, the Fourth Amendment —
where the NLA never saw the SAE.

📄 **Full write-up: [PAPER.md](PAPER.md)** (or the compilable [`paper/paper.tex`](paper/paper.tex)).

## Reproduce

Everything runs serverlessly on [Modal](https://modal.com) on one H100
(~10 min, ≈ $0.70, within the monthly free credits).

```bash
pip install modal && modal setup
modal run code/probe.py          # cheap integration check (2 activations)
modal run code/run.py            # full experiment -> results/results.json
python code/make_figures.py      # figures from results.json
python code/regenerate_report.py # browsable HTML report
```

## Layout

```
PAPER.md                 the paper (Markdown, renders on GitHub)
paper/paper.tex          LaTeX source (+ refs.bib) for a PDF
code/run.py              full experiment (Modal)
code/probe.py            3-stage integration probe (SAE / extraction / SGLang AV)
code/make_figures.py     figures from results.json
code/regenerate_report.py  HTML report from results.json
results/results.json     raw metrics + 216 per-token records
results/report.html      browsable report
figures/                 fig_agreement.png, fig_fidelity.png
```

## Method in one paragraph

For each token we take, from a single forward pass, the SAE at
`hidden_states[20]` (`resid_post_layer_19`) and the NLA at `hidden_states[21]`
("layer 20") — one block apart. The SAE gives a sparse feature code; the NLA
gives an English description, which we embed with `all-MiniLM-L6-v2`. Test A
asks whether tokens that fire a shared SAE feature have more-similar NLA
descriptions than random pairs (with a shuffled control). Test B correlates the
two induced similarity geometries. All comparisons are **label-free** — no human
names for SAE features are needed. See [PAPER.md](PAPER.md) for the caveats
(small N, the one-block layer offset, the MiniLM proxy, single-permutation
controls).

## Credits

The NLA (Anthropic / [kitft](https://github.com/kitft/natural_language_autoencoders)),
the SAE ([andyrdt](https://huggingface.co/andyrdt/saes-qwen2.5-7b-instruct)),
and the model (Qwen) are the work of their respective authors; this repo only
compares them. Code and analysis released under the [MIT License](LICENSE).
