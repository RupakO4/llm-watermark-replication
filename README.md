# Replicating "A Watermark for Large Language Models" on new data

COMP8240 (Applications of Data Science), Macquarie University, Session 2 2026. Rupak Pantha (60824670).
The final report is [`report/main.pdf`](report/main.pdf), with its LaTeX source in `report/`.

This repository replicates the soft "green-list" watermark of
[Kirchenbauer et al. (ICML 2023)](https://proceedings.mlr.press/v202/kirchenbauer23a.html) with the authors' own code
([jwkirchenbauer/lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking), pinned at commit `8292251`) and
the paper's own models. It then tests whether the paper's claims hold on nine kinds of text the paper never used and
with a modern generator. A pilot ran on a laptop CPU (v1, the git history before the v2 commit); the final study (v2)
ran on one Kaggle GPU:

| | pilot (v1, laptop CPU) | final (v2, one Kaggle GPU) |
|---|---|---|
| Generator | OPT-350M | **OPT-1.3B** (paper) |
| Perplexity oracle | OPT-1.3B | **OPT-2.7B** (paper) |
| Replication prompts (C4) | 100 | **500** (paper: ~500) |
| New-data prompts | 30 x 10 sources = 300 | **100 x 10 sources = 1,000** |
| T5 span attack | T5-Base, 5 beams, k=5, 50 texts | **T5-Large, 50 beams, k=20** (paper), 200 texts |
| LLM judge | Qwen2.5-1.5B-Instruct | **Qwen2.5-3B-Instruct** |
| Modern generator | - | **Qwen2.5-1.5B** on the same 1,000 prompts (oracle Qwen2.5-3B) |

Everything else is identical: the authors' watermarking code at commit `8292251`, LeftHash (`simple_1`) seeding,
multinomial sampling at temperature 0.7, z > 4, the same ten sources and preprocessing, the same cheap attacks, and
green lists drawn on the CPU (so a text watermarked on the GPU is detected by the same CPU detector as in v1).

## Headline results (report: `report/main.pdf`)

* **Replication (500 C4 prompts, OPT-1.3B).** TPR at z>4 is 0.988 at gamma=0.5, delta=2 (paper 0.984) and 0.986 at
  gamma=0.25 (paper 0.994). No human continuation is flagged. Two unwatermarked texts reach z>4 at gamma=0.25, and
  both drop below 2 when repeated bigrams are counted once. Only gamma=0.5, delta=1 falls short (0.654 vs 0.767).
* **T5-Large attack.** TPR is 0.96 / 0.61 / 0.20 / 0.07 at epsilon = 0.1 / 0.3 / 0.5 / 0.7 (paper 0.82 / 0.35 / 0.09 / 0.04).
  With repeats counted once it is 0.91 / 0.32 / 0.01 / 0.01.
* **Device dependence.** Text watermarked with CPU green lists scores mean z 7.94 on CPU and 0.04 on CUDA.
* **New data (1,000 prompts, 10 sources).** TPR at delta=2 is 0.98. Raw scoring flags 7 of 1,800 prose negatives
  (degenerate loops, one ethics statement) and 22 of 200 code negatives. Counting repeats once gives 0 false positives,
  but only 0.34 of watermarked code is then detected. Spearman rho between spike entropy and repeat-discounted z is 0.81.
* **Modern model (Qwen2.5-1.5B, same prompts).** TPR at delta=2 is 0.95, with 3 raw false positives and 0 with repeats
  counted once. Code TPR after deduplication is 0.55; the weak spots are Wikipedia and code.
* **Quality.** Oracle PPL rates human text worst, and the Qwen2.5-3B judge rates it best. Delta=2 costs about 0.1
  judge points and delta=5 two to five times as much.
* **Second annotator.** Two humans agree at weighted kappa 0.46 on 59 pilot texts; the 1.5B judge matches that
  ceiling (0.46 and 0.50), the 3B judge does not (0.16 and 0.15).
* **Judge check.** On the 59 pilot texts, the 3B judge ranks texts about as well as the pilot's 1.5B judge
  (Spearman 0.29 vs 0.34 with the author) but on a harsher scale (mean 2.42 vs the author's 3.46). The human ratings
  confirm the delta and perplexity findings on their own.
* **Screening.** 18 model outputs (34 rows with edited copies) that were sexually explicit, described sexual assault
  graphically or sexualised teenagers had their text deleted; their scores are kept (`data/removed_texts.csv`).

## Reproducing

Check every number in the report without running a model (about a minute):

```bash
pip install -r requirements.txt
python src/analysis.py --root release                  # every table, figure and number -> release/results/
cd report && RESULTS=../release/results ./build.sh     # compile the PDF from them
```

Rerun the experiments: upload `notebook/llm_watermark_v2.ipynb` to Kaggle (GPU + Internet on) or Colab and run all
cells. The notebook carries `src/` and `run_gpu.sh`, installs the library versions in `results/environment.json`,
fetches the authors' code at the pinned commit and runs `run_gpu.sh full` (about 9 hours on one GPU). Every stage
is resumable and stops cleanly before the platform's time limit. On a machine with a GPU:

```bash
bash setup.sh                    # authors' code @ 8292251 into third_party/, requirements
bash run_gpu.sh smoke            # every stage on 1-2 prompts per source
bash run_gpu.sh full             # the experiments
PART=modern bash run_gpu.sh full # one part only: rep | md | modern
python src/remove_texts.py       # delete the text of screened-out outputs (data/removed_texts.csv)
python src/make_release.py       # redacted public copy -> release/
python src/analysis.py           # tables, figures and numbers -> results/
```

| Folder | Contents |
|---|---|
| `src/` | all code: prompts, generation + detection, device check, perplexity/entropy, LLM judge, T5 attack, analysis |
| `release/` | public copy of the prompts, generations, scores, judge ratings and human ratings (CC-News, IMDb and Amazon text redacted) |
| `data/` | data card and per-source preprocessing reports |
| `results/` | tables, figures, `summary.json` and `environment.json` used by the report |
| `annotation_v1/` | the pilot texts rated blind by two annotators: answer key, both ratings, both judges' scores, excluded item (no text) |
| `notebook/` | the Kaggle/Colab notebook and the scripts that build it |

The full prompt sets and raw results contain source text that may not be redistributed, so they are not in the
repository; `release/` holds the same files with that text replaced by identifiers, and `src/build_prompts.py`
rebuilds the originals deterministically. The judge check of the 60 rated texts (`notebook/make_judge_check.py`)
needs the private copy of those texts and is therefore not runnable from the public repository; its output is in
`release/results/v1_items/judge.csv`.

What changed in the code relative to v1:

* `wm_common.generate_batch` / `RowGreenLists`: one left-padded `generate()` call produces the unwatermarked text
  and every (gamma, delta) watermarked text for several prompts; each row's green list is still drawn on CPU by
  the authors' processor. `generate.py --batch_prompts N` uses it.
* `retokenize_prompts.py`: re-encodes the prompt sets for a generator with another tokenizer (Qwen2.5).
* `quality.py`: trims the oldest prompt tokens if prompt + continuation exceed the model's context.
* `analysis.py`: compares the two generators on the same prompts (`modern_compare.tex`, `mod.*` numbers).
* `run_gpu.sh`: the v2 stages and models; `WM_DEADLINE` stops any stage cleanly at the time budget.
* The 60 texts rated by the author in v1 (`embed/`) are re-rated by the v2 judge, to validate it against the human.

## The device rule

The green list for each token is a `torch.randperm` drawn from a `torch.Generator` seeded with the previous token. CPU,
MPS and CUDA generators produce different permutations for the same seed, so all green lists here are drawn on the
**CPU**, also when the model runs on the GPU. This makes the key portable.

## Licences

Code: MIT (see `LICENSE`). The authors' watermarking code is Apache-2.0 and is fetched, not vendored. Data: every
prompt records its source, row index and licence. `release/` redistributes source text only from openly licensed
sources (Wikipedia and Stack Exchange: CC BY-SA; ACL Anthology papers and the Reddit dump: CC BY; C4: ODC-BY;
Gutenberg: public domain; Python: PSF). All generated continuations are machine-written by OPT-1.3B or Qwen2.5-1.5B.

## Use of AI

Generative AI (Claude) was used to help write code and edit the report, as the unit's AI policy allows. All results
come from the code in this repository, and every reference was checked by the author.
