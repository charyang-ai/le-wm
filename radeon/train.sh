#!/bin/bash
# Inside the container: train LeWM from scratch on one card with the paper's schedule (10 epochs, appendix; the
# repo default is 100), then evaluate the last checkpoint with the upstream eval config.
#   bash radeon/train.sh tworoom lewm_tworoom_w7900
# - Push-T ships on HF as pusht_expert_train.h5 while config/train/data/pusht.yaml names a .lance table.
# - Pixels are blosc-compressed in 100-frame chunks; 24 loader workers instead of 6 keep the GPU fed.
# - TunableOp times the candidate bf16 GEMM kernels on this card once and caches the choice
#   (radeon/bench_train_step.py: 440 -> 279 ms per synthetic step).
set -uo pipefail
data=${1:?tworoom|pusht|dmc|ogb}
run=${2:-lewm_${data}_w7900}
shift 2 || shift $#
cd "$(dirname "$0")/.." || exit 1
R=${RESULTS:-/workspace/results}
mkdir -p "$R/train" "$R/eval" "$R/tunableop"
case "$data" in
  pusht) extra=(data.dataset.name=pusht_expert_train.h5) ;;
  *) extra=() ;;
esac
export PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 PYTORCH_TUNABLEOP_FILENAME="$R/tunableop/lewm.csv"
log="$R/train/$run.log"
echo "=== train $data $run $(date -Is)" | tee -a "$log"
python train.py data="$data" "${extra[@]}" trainer.max_epochs=10 num_workers=24 output_model_name="$run" "$@" 2>&1 \
  | tee -a "$log"
echo "exit ${PIPESTATUS[0]} $(date -Is)" | tee -a "$log"
declare -A EVAL=([tworoom]=tworoom [pusht]=pusht [dmc]=reacher [ogb]=cube)
python eval.py --config-name="${EVAL[$data]}.yaml" policy="$run/weights_epoch_10.pt" > "$R/eval/${run}.log" 2>&1
grep -E "success_rate|Error" "$R/eval/${run}.log" | cut -c1-80 | tail -n 2
