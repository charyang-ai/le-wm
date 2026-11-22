#!/bin/bash
# Inside the container: the released LeWM checkpoints and datasets (Hugging Face collection quentinll/lewm), laid out
# under $STABLEWM_HOME the way stable-worldmodel 0.1.1 and the le-wm configs look them up:
#   checkpoints/models--quentinll--lewm-<env>/{config.json,weights.pt}   -> policy=quentinll/lewm-<env>
#   datasets/tworoom.h5, pusht_expert_train.h5, reacher.h5 (+ dmc/reacher_random.h5), ogbench/cube_single_expert.h5
#   bash /workspace/src/radeon/download.sh tworooms pusht reacher cube
# load_pretrained() downloads from a hard-coded huggingface.co unless the checkpoint folder exists, so it is filled
# here with the hf CLI, which honours HF_ENDPOINT. Mirrors rate-limit API calls (HTTP 429), hence the retries.
set -uo pipefail
D=${STABLEWM_HOME:?set STABLEWM_HOME}
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60
retry() {
  for i in 1 2 3 4 5 6 7 8; do
    "$@" && return 0
    echo "attempt $i failed, retrying in 330 s: $*"
    sleep 330
  done
  return 1
}
fetch_model() {
  local dir="$D/checkpoints/models--quentinll--lewm-$1"
  hf download "quentinll/lewm-$1" config.json weights.pt --local-dir "$dir"
  # On a 429 the CLI can fall back to whatever is already in --local-dir and still exit 0.
  [ -s "$dir/config.json" ] && [ -s "$dir/weights.pt" ]
}
mkdir -p "$D/checkpoints" "$D/datasets/dmc" "$D/datasets/ogbench" "$D/_zst"
for env in "$@"; do
  echo "=== $env $(date -Is)"
  retry fetch_model "$env" || exit 1
  retry hf download "quentinll/lewm-$env" --repo-type dataset --local-dir "$D/_zst/$env" || exit 1
  for z in "$D/_zst/$env"/*.zst; do
    [ -f "$z.unpacked" ] && continue
    case "$z" in
      *.tar.zst) tar --zstd -xvf "$z" -C "$D/datasets" ;;
      *.zst)     zstd -d -f "$z" -o "$D/datasets/$(basename "${z%.zst}")" ;;
    esac && touch "$z.unpacked" || exit 1
  done
done
# config/eval asks for dmc/reacher_random and ogbench/cube_single_expert; the archives hold reacher.h5 and
# cube_single_expert.h5 (config/train/data reads reacher.h5 and ogbench/cube_single_expert.h5).
[ -f "$D/datasets/reacher.h5" ] && ln -sfn ../reacher.h5 "$D/datasets/dmc/reacher_random.h5"
[ -f "$D/datasets/cube_single_expert.h5" ] && mv -f "$D/datasets/cube_single_expert.h5" "$D/datasets/ogbench/"
find "$D/checkpoints" "$D/datasets" \( -name '*.h5' -o -name '*.pt' -o -name 'config.json' \) -printf '%14s  %p\n'
