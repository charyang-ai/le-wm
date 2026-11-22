#!/bin/bash
# Inside the container: system libraries for MuJoCo's EGL rendering, then stable-worldmodel with the pins this port
# needs. torch and torchvision from the image are held fixed with a constraints file.
#   docker exec radeon-le-wm bash /workspace/src/radeon/install.sh
set -euo pipefail
LOG=${LOG:-/workspace/logs}
mkdir -p "$LOG"
exec > >(tee -a "$LOG/install.log") 2>&1
export DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
if [ -n "${APT_MIRROR:-}" ]; then
  sed -i -E "s#https?://(archive|security)\.ubuntu\.com/ubuntu/?#${APT_MIRROR}#g" /etc/apt/sources.list.d/ubuntu.sources
fi
apt-get update
apt-get install -y --no-install-recommends libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libgles2 libglvnd0 libosmesa6 \
  zstd ffmpeg git curl
python - <<'PY' > "$LOG/torch_constraints.txt"
import torch, torchvision
print(f"torch=={torch.__version__}")
print(f"torchvision=={torchvision.__version__}")
PY
# - "format": since stable-worldmodel 0.1.1 the HDF5 reader eval.py uses lives in this extra; the le-wm README only
#   installs [train,env].
# - transformers<5: 5.x renamed the ViT parameters the released checkpoints were saved with. transformers 4 needs
#   huggingface-hub<1, which needs datasets<5.
# - mujoco/dm_control: from MuJoCo 3.12 (dm_control 1.0.45) on, the DMC Reacher floor no longer looks like the frames
#   in the released datasets (radeon/render_check.py); 3.11.0 / 1.0.44 is the last pair that matches.
python -m pip install -c "$LOG/torch_constraints.txt" \
  "stable-worldmodel[train,env,format]==0.1.1" "transformers<5" "datasets<5" "mujoco==3.11.0" "dm_control==1.0.44"
python -m pip freeze > "$LOG/pip_freeze.txt"
python -c "import torch, transformers, mujoco; print('torch', torch.__version__, 'transformers', transformers.__version__, 'mujoco', mujoco.__version__)"
