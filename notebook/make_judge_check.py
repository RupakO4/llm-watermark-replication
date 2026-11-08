#!/usr/bin/env python3
"""Build judge_check.ipynb: re-rate the 60 texts the author rated in v1 with the v2 judge.

About 10 minutes on a Colab/Kaggle T4. Output: judge_v1_items.csv (download it).
    python notebook/make_judge_check.py
"""
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebook" / "judge_check.ipynb"

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for p in sorted((ROOT / "src").glob("*.py")):
        tar.add(p, arcname=f"src/{p.name}")
    tar.add(ROOT / "embed" / "data" / "prompts_v1_items.csv", arcname="data/prompts_v1_items.csv")
    tar.add(ROOT / "embed" / "results" / "v1_items" / "generations.csv", arcname="results/v1_items/generations.csv")
CODE = base64.b64encode(buf.getvalue()).decode()


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)}


cells = [
    md("""
# Judge check: re-rate the 60 human-rated texts with the v2 judge (Qwen2.5-3B-Instruct)

**Colab:** *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. About 10 minutes.
At the end the browser downloads **judge_v1_items.csv**; put it in the `llm-watermark-v2` folder.

**Kaggle:** GPU + Internet on, *Run all*; the file appears under *Output*.
"""),
    code("""
import os, sys, io, base64, tarfile, subprocess, shutil
W = "/kaggle/working/judge_check" if os.path.isdir("/kaggle/working") else "/content/judge_check"
os.makedirs(W, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode("__CODE__")), mode="r:gz") as t:
    t.extractall(W)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.16.1", "tokenizers==0.23.1", "scipy"])
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - switch the runtime to a GPU")
"""),
    code("""
r = subprocess.run([sys.executable, "src/llm_judge.py", "--experiment", "v1_items", "--prompts", "data/prompts_v1_items.csv",
                    "--judge", "Qwen/Qwen2.5-3B-Instruct", "--device", "auto"], cwd=W, text=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(r.stdout[-3000:])
out = f"{W}/results/v1_items/judge.csv"
dst = ("/kaggle/working" if os.path.isdir("/kaggle/working") else "/content") + "/judge_v1_items.csv"
shutil.copy(out, dst)
print("rows:", sum(1 for _ in open(dst)) - 1, "->", dst)
try:
    from google.colab import files
    files.download(dst)
except ImportError:
    pass
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "T4"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1).replace("__CODE__", CODE), encoding="utf-8")
print("wrote", OUT)
