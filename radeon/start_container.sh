#!/bin/bash
# Start the one-GPU ROCm container. The work dir is mounted at /workspace and this repo is expected at $WORK/src.
#   bash radeon/start_container.sh /path/to/work /dev/dri/renderD129
set -euo pipefail
WORK=${1:?work dir}
NODE=${2:?render node of the one GPU to use, e.g. /dev/dri/renderD129}
NAME=${NAME:-radeon-le-wm}
IMAGE=${IMAGE:-rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0}
mkdir -p "$WORK"/{data,hf,logs,results}
docker run -d --name "$NAME" \
  --device=/dev/kfd --device="$NODE" \
  --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
  --ipc=host --shm-size=32g --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0 \
  -e STABLEWM_HOME=/workspace/data -e HF_HOME=/workspace/hf -e HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
  -v "$WORK":/workspace -w /workspace/src \
  --entrypoint bash "$IMAGE" -lc 'sleep infinity'
docker exec "$NAME" python -c "import torch; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
