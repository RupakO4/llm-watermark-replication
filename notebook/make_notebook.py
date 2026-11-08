#!/usr/bin/env python3
"""Build llm_watermark_v2.ipynb: one self-contained notebook for Kaggle or Colab.

The project code (src/, run_gpu.sh) is embedded as a base64 tar.gz, so the notebook
is the only file that has to be uploaded.  Run from the v2 folder:
    python notebook/make_notebook.py
"""
import base64
import io
import sys
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebook" / "llm_watermark_v2.ipynb"

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for p in sorted((ROOT / "src").glob("*.py")):
        tar.add(p, arcname=f"src/{p.name}")
    for name in ("run_gpu.sh", "requirements.txt"):
        tar.add(ROOT / name, arcname=name)
    # the 60 texts rated by the author in v1, for re-judging with the v2 judge (private: contains
    # text from sources that may not be redistributed, so the public notebook is built with --public)
    if "--public" not in sys.argv and (ROOT / "embed").exists():
        tar.add(ROOT / "embed" / "data" / "prompts_v1_items.csv", arcname="data/prompts_v1_items.csv")
        tar.add(ROOT / "embed" / "results" / "v1_items" / "generations.csv", arcname="results/v1_items/generations.csv")
CODE = base64.b64encode(buf.getvalue()).decode()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


cells = [
    md("""
# Watermark replication v2: the paper's models, the paper's scale, plus a modern model

Runs the whole v2 experiment on one GPU and saves everything as it goes.

* **Replication**: OPT-1.3B generator, 500 C4 prompts, γ ∈ {0.25, 0.5} × δ ∈ {1, 2, 5}, OPT-2.7B perplexity oracle,
  T5-Large span attack (50 beams, k = 20); the paper's own models.
* **New data**: 100 prompts from each of 10 sources (1,000 prompts) with OPT-1.3B.
* **Modern model**: the same 1,000 prompts with Qwen2.5-1.5B (oracle Qwen2.5-3B).
* **Judge**: Qwen2.5-3B-Instruct rates every text 1-5.

Green lists are always drawn on the CPU, so the texts are detectable by the same CPU detector as the laptop run.

### Kaggle
1. *Create → New Notebook → File → Import Notebook*, upload this file.
2. Right-hand panel → *Session options*: **Accelerator = GPU T4 x2** (or P100) and **Internet = On**
   (Internet needs a phone-verified account).
3. *Save Version → Save & Run All (Commit)*. It runs in the background (laptop can be closed) for up to 11 h.
4. When it finishes: open the version → *Output* → download **v2_results.zip**.
   If it stopped at the time limit: *Add Input → Notebook Output →* this notebook, then *Save & Run All* again; it resumes.

### Colab
1. *File → Upload notebook*, choose this file. *Runtime → Change runtime type → T4 GPU*.
2. *Runtime → Run all*. Allow Google Drive access; results are saved in **My Drive/llm-watermark-v2** as it goes.
3. Keep the tab open. If Colab disconnects, reconnect and *Run all* again: finished work is skipped.
4. At the end, **v2_results.zip** is in My Drive.
"""),
    code("""
# ---- settings ----------------------------------------------------------------------------------
PART = "all"              # "all", or any of "rep", "md", "modern" (e.g. "rep md")
MODE = "full"             # "smoke" = every stage on 1-2 prompts per source (quick test)
TIME_BUDGET_HOURS = 11.0  # stop cleanly before the platform limit (Kaggle 12 h); rerun to resume
"""),
    code("""
# ---- setup: platform, code, dependencies --------------------------------------------------------
import os, sys, time, base64, io, tarfile, subprocess, shutil, glob
START = time.time()
ON_KAGGLE = os.path.isdir("/kaggle/working")
ON_COLAB = (not ON_KAGGLE) and ("google.colab" in sys.modules or os.path.isdir("/content"))
if ON_KAGGLE:
    BASE = "/kaggle/working/llm-watermark-v2"
elif ON_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = "/content/drive/MyDrive/llm-watermark-v2"
else:
    BASE = os.path.abspath("llm-watermark-v2")
os.makedirs(BASE, exist_ok=True)
print("platform:", "Kaggle" if ON_KAGGLE else "Colab" if ON_COLAB else "local", "| folder:", BASE)

# unpack the project code (results and data already in the folder are kept)
CODE_B64 = "__CODE__"
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64)), mode="r:gz") as t:
    t.extractall(BASE)

# Kaggle resume: copy the results of an earlier run of this notebook attached as input
if ON_KAGGLE:
    for prev in glob.glob("/kaggle/input/**/llm-watermark-v2", recursive=True):
        for sub in ("data", "results", "annotation"):
            if os.path.isdir(f"{prev}/{sub}"):
                shutil.copytree(f"{prev}/{sub}", f"{BASE}/{sub}", dirs_exist_ok=True)
        subprocess.run(["chmod", "-R", "u+w", BASE])
        print("resuming from", prev)

# the authors' watermarking code, pinned to the commit used in the report
UPSTREAM = "/tmp/lm-watermarking"
if not os.path.isdir(UPSTREAM):
    subprocess.run(["git", "clone", "-q", "https://github.com/jwkirchenbauer/lm-watermarking.git", UPSTREAM], check=True)
subprocess.run(["git", "-C", UPSTREAM, "checkout", "-q", "82922516930c02f8aa322765defdb5863d07a00e"], check=True)

# dependencies: the platform's torch + the library versions of the laptop run (results/environment.json);
# upgrading datasets alone pulls a huggingface_hub that breaks older transformers, so they move together
PINNED = ["transformers==5.16.1", "datasets==5.1.0", "tokenizers==0.23.1"]
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINNED, "sentencepiece", "scikit-learn", "scipy"])
if r.returncode != 0:
    print("pinned versions failed; installing the latest compatible set instead")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "datasets", "tokenizers",
                    "huggingface_hub", "sentencepiece", "scikit-learn", "scipy"], check=True)
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("!!! NO GPU - switch the accelerator / runtime type to a GPU, then run again.")
    if MODE == "full":
        raise SystemExit("No GPU")
"""),
    code("""
# ---- run (prints progress; every stage resumes where it stopped) ---------------------------------
env = dict(os.environ, PART=PART, PYTHON=sys.executable, LMW_UPSTREAM=UPSTREAM,
           WM_DEADLINE=str(START + TIME_BUDGET_HOURS * 3600), TOKENIZERS_PARALLELISM="false")
p = subprocess.Popen(["bash", "run_gpu.sh", MODE], cwd=BASE, env=env, text=True, bufsize=1,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for line in p.stdout:
    print(line, end="")
p.wait()
print(f"finished in {(time.time() - START) / 3600:.2f} h")
"""),
    code("""
# ---- what has been done so far ---------------------------------------------------------------------
import csv
csv.field_size_limit(10**9)
for f in sorted(glob.glob(f"{BASE}/results/*/*.csv")):
    with open(f, encoding="utf-8") as fh:
        n = sum(1 for _ in csv.reader(fh)) - 1
    print(f"{f[len(BASE) + 1:]:55s} {n:7d} rows")
for f in sorted(glob.glob(f"{BASE}/results/logs/*.log")):
    if "!!!" in open(f, encoding="utf-8", errors="ignore").read():
        print("problem in", f)
"""),
    code("""
# ---- package the results ------------------------------------------------------------------------
zip_base = "/kaggle/working/v2_results" if ON_KAGGLE else (
    "/content/drive/MyDrive/v2_results" if ON_COLAB else os.path.abspath("v2_results"))
tmp = "/tmp/v2_pack/llm-watermark-v2"
shutil.rmtree("/tmp/v2_pack", ignore_errors=True)
for sub in ("data", "results", "annotation"):
    if os.path.isdir(f"{BASE}/{sub}"):
        shutil.copytree(f"{BASE}/{sub}", f"{tmp}/{sub}", ignore=shutil.ignore_patterns("cache"))
print(shutil.make_archive(zip_base, "zip", "/tmp/v2_pack"))
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "accelerator": "GPU", "colab": {"gpuType": "T4"}},
      "nbformat": 4, "nbformat_minor": 5}
text = json.dumps(nb, indent=1, ensure_ascii=False).replace("__CODE__", CODE)
OUT.write_text(text, encoding="utf-8")
print(f"wrote {OUT} ({len(text) / 1024:.0f} KB)")
