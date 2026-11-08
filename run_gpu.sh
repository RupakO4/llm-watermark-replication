#!/usr/bin/env bash
# v2: the paper's models at the paper's scale, plus a modern generator, on one GPU (Kaggle / Colab).
# Every stage is resumable: rerun the script and finished prompts are skipped.
#
#   bash run_gpu.sh full            # everything (~5-7 h on a T4)
#   bash run_gpu.sh smoke           # 1-2 prompts per source through every stage (~20 min + downloads)
#   PART=rep bash run_gpu.sh full   # only one part: rep | md | modern | all (default)
#
# Models and sizes can be overridden through the environment (used for offline tests).
set -uo pipefail
cd "$(dirname "$0")"
MODE="${1:-smoke}"
PY="${PYTHON:-python}"
PART="${PART:-all}"
export PYTHONUNBUFFERED=1

GEN="${GEN_MODEL:-facebook/opt-1.3b}"            # paper's generator
ORACLE="${ORACLE_MODEL:-facebook/opt-2.7b}"      # paper's perplexity oracle
MODERN="${MODERN_MODEL:-Qwen/Qwen2.5-1.5B}"      # modern generator (base model)
MODERN_ORACLE="${MODERN_ORACLE:-Qwen/Qwen2.5-3B}" # larger model of the same family
JUDGE="${JUDGE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
T5="${T5_MODEL:-google-t5/t5-large}"             # paper's attacker
T5_BEAMS="${T5_BEAMS:-50}"; T5_K="${T5_K:-20}"   # paper: 50-way beam search, k=20 candidates
PROMPT_TOK="${PROMPT_TOKENIZER:-facebook/opt-1.3b}"
DEV="${DEVICE:-auto}"
DTYPE="${DTYPE:-float16}"

if [[ "$MODE" == "smoke" ]]; then
  REP=smoke_replication; MD=smoke_multidomain; N_REP=2; N_MD=1
  T5_REP_LIMIT=2; T5_MD_LIMIT=1; B_REP=2; B_MD=4
elif [[ "$MODE" == "full" ]]; then
  REP=replication; MD=multidomain; N_REP="${N_REP:-500}"; N_MD="${N_MD:-100}"
  T5_REP_LIMIT="${T5_REP_LIMIT:-200}"; T5_MD_LIMIT="${T5_MD_LIMIT:-10}"
  B_REP="${B_REP:-4}"; B_MD="${B_MD:-16}"
else
  echo "usage: bash run_gpu.sh [smoke|full]"; exit 1
fi
QW="${MD}_qwen"
mkdir -p results/logs

has() { [[ "$PART" == "all" || " $PART " == *" $1 "* ]]; }
run() {
  [[ -f ABORT ]] && { echo "ABORT file found"; exit 1; }
  if [[ -n "${WM_DEADLINE:-}" ]] && (( $(date +%s) > ${WM_DEADLINE%.*} )); then
    echo "=== time budget used up - skipping $1"; return 0; fi
  local name="$1"; shift
  echo; echo "=== [$MODE] $name  $(date '+%H:%M:%S')"
  "$@" 2>&1 | tee "results/logs/${MODE}_${name}.log"
  local rc=${PIPESTATUS[0]}
  [[ $rc -ne 0 ]] && echo "!!! stage $name exited with code $rc (continuing with the next stage)"
  return 0
}

run env $PY src/env_report.py

# ---- data ------------------------------------------------------------------------------------
if has rep && [[ ! -f data/prompts_$REP.csv ]]; then
  run prompts_rep $PY src/build_prompts.py --set replication --name $REP --per_source $N_REP --tokenizer $PROMPT_TOK
fi
if (has md || has modern) && [[ ! -f data/prompts_$MD.csv ]]; then
  run prompts_md $PY src/build_prompts.py --set multidomain --name $MD --per_source $N_MD --tokenizer $PROMPT_TOK
fi

# ---- 1. replication: OPT-1.3B, 500 C4 prompts, gamma x delta grid --------------------------------
if has rep; then
  run gen_rep $PY src/generate.py --experiment $REP --prompts data/prompts_$REP.csv --model $GEN \
      --gammas 0.25 0.5 --deltas 1 2 5 --max_new_tokens 200 --model_device $DEV --dtype $DTYPE --batch_prompts $B_REP
  run device $PY src/device_check.py --experiment $REP --prompts data/prompts_$REP.csv --model $GEN
  run quality_rep $PY src/quality.py --experiment $REP --prompts data/prompts_$REP.csv \
      --oracle $ORACLE --generator $GEN --device $DEV
  run judge_rep $PY src/llm_judge.py --experiment $REP --prompts data/prompts_$REP.csv --gamma 0.5 \
      --judge $JUDGE --device $DEV
  run t5_rep $PY src/attack_t5.py --experiment $REP --prompts data/prompts_$REP.csv --model $GEN \
      --attack_model $T5 --num_beams $T5_BEAMS --k $T5_K --device $DEV \
      --budgets 0.1 0.3 0.5 0.7 --limit_per_source $T5_REP_LIMIT
  run quality_rep_t5 $PY src/quality.py --experiment $REP --prompts data/prompts_$REP.csv \
      --oracle $ORACLE --generator $GEN --device $DEV --skip_entropy
fi

# ---- 2. new data: OPT-1.3B, 100 prompts from each of 10 sources ----------------------------------
if has md; then
  run gen_md $PY src/generate.py --experiment $MD --prompts data/prompts_$MD.csv --model $GEN \
      --gammas 0.5 --deltas 1 2 5 --max_new_tokens 150 --model_device $DEV --dtype $DTYPE --batch_prompts $B_MD
  run quality_md $PY src/quality.py --experiment $MD --prompts data/prompts_$MD.csv \
      --oracle $ORACLE --generator $GEN --device $DEV
  run judge_md $PY src/llm_judge.py --experiment $MD --prompts data/prompts_$MD.csv --judge $JUDGE --device $DEV
  run t5_md $PY src/attack_t5.py --experiment $MD --prompts data/prompts_$MD.csv --model $GEN \
      --attack_model $T5 --num_beams $T5_BEAMS --k $T5_K --device $DEV \
      --budgets 0.1 0.3 0.5 --limit_per_source $T5_MD_LIMIT
  run quality_md_t5 $PY src/quality.py --experiment $MD --prompts data/prompts_$MD.csv \
      --oracle $ORACLE --generator $GEN --device $DEV --skip_entropy
  [[ "$MODE" == "full" ]] && run annotate $PY src/make_annotation.py --experiment $MD --prompts data/prompts_$MD.csv
  # the 60 texts the author rated in v1: re-judged with the v2 judge, so it can be checked against the human
  [[ -f data/prompts_v1_items.csv ]] && run judge_v1 $PY src/llm_judge.py --experiment v1_items \
      --prompts data/prompts_v1_items.csv --judge $JUDGE --device $DEV
fi

# ---- 3. the same new data with a modern generator ------------------------------------------------
if has modern; then
  [[ -f data/prompts_$QW.csv ]] || run retok $PY src/retokenize_prompts.py --src data/prompts_$MD.csv \
      --dst data/prompts_$QW.csv --tokenizer $MODERN
  run gen_qwen $PY src/generate.py --experiment $QW --prompts data/prompts_$QW.csv --model $MODERN \
      --gammas 0.5 --deltas 1 2 5 --max_new_tokens 150 --model_device $DEV --dtype $DTYPE \
      --batch_prompts $B_MD --no_attacks
  run quality_qwen $PY src/quality.py --experiment $QW --prompts data/prompts_$QW.csv \
      --oracle $MODERN_ORACLE --generator $MODERN --device $DEV
  run judge_qwen $PY src/llm_judge.py --experiment $QW --prompts data/prompts_$QW.csv --judge $JUDGE --device $DEV
  run t5_qwen $PY src/attack_t5.py --experiment $QW --prompts data/prompts_$QW.csv --model $MODERN \
      --attack_model $T5 --num_beams $T5_BEAMS --k $T5_K --device $DEV \
      --budgets 0.1 0.3 0.5 --limit_per_source $T5_MD_LIMIT
  run quality_qwen_t5 $PY src/quality.py --experiment $QW --prompts data/prompts_$QW.csv \
      --oracle $MODERN_ORACLE --generator $MODERN --device $DEV --skip_entropy
fi
echo; echo "=== [$MODE] finished part '$PART' $(date '+%H:%M:%S')"
