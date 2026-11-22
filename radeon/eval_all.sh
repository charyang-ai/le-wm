#!/bin/bash
# Inside the container: the upstream planning evaluation (eval.py, 50 episodes, CEM 300 samples x 30 iterations)
# with the released checkpoint and with the random policy, one log per run in results/eval/.
#   bash radeon/eval_all.sh tworoom pusht reacher cube
#   SEEDS="42 1 2" bash radeon/eval_all.sh reacher
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
R=${RESULTS:-/workspace/results}/eval
mkdir -p "$R"
declare -A HF=([tworoom]=tworooms [pusht]=pusht [reacher]=reacher [cube]=cube)
for cfg in "$@"; do
  for seed in ${SEEDS:-42}; do
    tag=$cfg
    [ "$seed" = 42 ] || tag=${cfg}_seed$seed
    for policy in "quentinll/lewm-${HF[$cfg]}" random; do
      log="$R/$tag.log"
      [ "$policy" = random ] && log="$R/${tag}_random.log"
      echo "=== $cfg seed=$seed $policy $(date -Is)"
      python eval.py --config-name="$cfg.yaml" policy="$policy" seed="$seed" > "$log" 2>&1
      echo "exit $? $(date -Is)"
      grep -E "success_rate|Error" "$log" | cut -c1-80 | tail -n 2
    done
  done
done
