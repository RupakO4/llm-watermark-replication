#!/usr/bin/env bash
# Run every experiment in the report, in order.  Each stage is resumable: if the
# run is interrupted, start it again and finished prompts are skipped.
#
#   bash run_all.sh smoke     # ~10-15 min + model downloads: 1-2 prompts per source, every stage
#   bash run_all.sh full      # the report's experiments (~9 h on an 8 GB M1 MacBook, CPU)
#   STAGES="gen_md quality_md" bash run_all.sh full     # run selected stages only
#
# Stages: env prompts gen_rep device gen_md quality_rep quality_md judge_rep judge_md annotate t5_rep t5_md
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-smoke}"
PY="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
ALL="env prompts gen_rep device gen_md quality_rep quality_md judge_rep judge_md annotate t5_rep t5_md"
STAGES="${STAGES:-$ALL}"

if [[ "$MODE" == "smoke" ]]; then
  REP=smoke_replication; MD=smoke_multidomain
  N_REP=2; N_MD=1; T5_MD_LIMIT=1; T5_REP_LIMIT=2
elif [[ "$MODE" == "full" ]]; then
  REP=replication; MD=multidomain
  N_REP=100; N_MD=30; T5_MD_LIMIT=5; T5_REP_LIMIT=50
else
  echo "usage: bash run_all.sh [smoke|full]"; exit 1
fi

mkdir -p results/logs
# keep a Mac awake while this script runs
if command -v caffeinate >/dev/null 2>&1; then caffeinate -i -w $$ & fi

has() { [[ " $STAGES " == *" $1 "* ]]; }
run() { [[ -f ABORT ]] && { echo "ABORT file found"; exit 1; }; local name="$1"; shift; echo; echo "=== [$MODE] $name  $(date '+%H:%M:%S')"; \
        "$@" 2>&1 | tee "results/logs/${MODE}_${name}.log"; }

has env      && run env $PY src/env_report.py
# prompt sets are built once (streaming the corpora takes ~20 min); delete the CSV to rebuild
if has prompts; then
  [[ -f data/prompts_$REP.csv ]] || run prompts_rep $PY src/build_prompts.py --set replication --name $REP --per_source $N_REP
  [[ -f data/prompts_$MD.csv ]]  || run prompts_md  $PY src/build_prompts.py --set multidomain --name $MD --per_source $N_MD
fi
has gen_rep  && run gen_rep $PY src/generate.py --experiment $REP --prompts data/prompts_$REP.csv \
                  --gammas 0.25 0.5 --deltas 1 2 5 --max_new_tokens 200
has device   && run device  $PY src/device_check.py --experiment $REP --prompts data/prompts_$REP.csv
has gen_md   && run gen_md  $PY src/generate.py --experiment $MD --prompts data/prompts_$MD.csv \
                  --gammas 0.5 --deltas 1 2 5 --max_new_tokens 150
has quality_rep && run quality_rep $PY src/quality.py --experiment $REP --prompts data/prompts_$REP.csv
has quality_md  && run quality_md  $PY src/quality.py --experiment $MD --prompts data/prompts_$MD.csv
has judge_rep && run judge_rep $PY src/llm_judge.py --experiment $REP --prompts data/prompts_$REP.csv --gamma 0.5
has judge_md  && run judge_md  $PY src/llm_judge.py --experiment $MD --prompts data/prompts_$MD.csv
if has annotate && [[ "$MODE" == "full" ]]; then
  run annotate $PY src/make_annotation.py --experiment $MD --prompts data/prompts_$MD.csv
fi
# the T5 attack is the slowest stage on a laptop CPU, so it runs last; quality.py then
# scores only the attacked texts it has not seen yet
has t5_rep   && run t5_rep  $PY src/attack_t5.py --experiment $REP --prompts data/prompts_$REP.csv \
                  --budgets 0.1 0.3 0.5 0.7 --limit_per_source $T5_REP_LIMIT
has t5_md    && run t5_md   $PY src/attack_t5.py --experiment $MD --prompts data/prompts_$MD.csv \
                  --budgets 0.1 0.3 0.5 --limit_per_source $T5_MD_LIMIT
has t5_rep   && run quality_rep_t5 $PY src/quality.py --experiment $REP --prompts data/prompts_$REP.csv --skip_entropy
has t5_md    && run quality_md_t5  $PY src/quality.py --experiment $MD --prompts data/prompts_$MD.csv --skip_entropy
echo; echo "=== [$MODE] all requested stages finished $(date '+%H:%M:%S')"
