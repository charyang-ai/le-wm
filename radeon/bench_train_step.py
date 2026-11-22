"""Time LeWM training steps on synthetic Push-T-shaped batches (no data loading), then profile the slowest kernels.

python radeon/bench_train_step.py [steps] [batch]
"""

import os
import sys
import time

import hydra
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from module import SIGReg  # noqa: E402

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
B = int(sys.argv[2]) if len(sys.argv) > 2 else 128
T, A = 4, 10  # history 3 + 1 prediction; frameskip 5 x 2-D Push-T action

cfg = OmegaConf.load("config/train/model/lewm.yaml")
cfg = OmegaConf.merge(OmegaConf.create({"img_size": 224, "history_size": 3, "embed_dim": 192}), {"model": cfg})
cfg.model.action_encoder.input_dim = A
model = hydra.utils.instantiate(OmegaConf.to_container(cfg, resolve=True)["model"]).cuda().train()
sigreg = SIGReg(knots=17, num_proj=1024).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M")

batch = {"pixels": torch.randn(B, T, 3, 224, 224, device="cuda"), "action": torch.randn(B, T, A, device="cuda")}


def step():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.encode(dict(batch))
        emb, act = out["emb"], out["act_emb"]
        pred = model.predict(emb[:, :3], act[:, :3])
        loss = (pred - emb[:, 1:]).pow(2).mean() + 0.09 * sigreg(emb.transpose(0, 1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


for _ in range(3):
    step()
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(steps):
    step()
torch.cuda.synchronize()
dt = (time.perf_counter() - t) / steps
print(f"train step B={B}: {dt * 1000:.1f} ms  ({1 / dt:.2f} it/s)", flush=True)

from torch.profiler import ProfilerActivity, profile

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        step()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=70))
