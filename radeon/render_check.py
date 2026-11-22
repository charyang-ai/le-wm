"""Render dataset states in the swm env and compare with the stored dataset frames.

python render_check.py reacher [n_rows]
Writes $RESULTS/render_check/<env><RC_TAG>_*.png (dataset | ours | |diff|x4) and <env><RC_TAG>.json.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json
import sys
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

import stable_worldmodel as swm

ENVS = {
    "reacher": dict(h5="reacher.h5", world=dict(env_name="swm/ReacherDMControl-v0", task="qpos_match")),
    "cube": dict(h5="ogbench/cube_single_expert.h5",
                 world=dict(env_name="swm/OGBCube-v0", env_type="single", ob_type="states", multiview=False,
                            width=224, height=224, visualize_info=False, terminate_at_goal=True)),
}


def main() -> None:
    name = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    tag = os.environ.get("RC_TAG", "")
    spec = ENVS[name]
    out = Path(os.environ.get("RESULTS", "/workspace/results"), "render_check")
    out.mkdir(parents=True, exist_ok=True)

    world = swm.World(num_envs=1, image_shape=(224, 224), max_episode_steps=100, **spec["world"])
    world.reset(seed=0)
    env = world.envs.envs[0].unwrapped

    f = h5py.File(Path(swm.data.utils.get_cache_dir(), "datasets", spec["h5"]), "r")
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(f["pixels"].shape[0], size=n, replace=False))
    report = {"env": name, "versions": {p: version(p) for p in ("mujoco", "dm_control", "stable-worldmodel")},
              "mujoco_gl": os.environ["MUJOCO_GL"], "rows": []}
    for r in rows:
        env.set_state(f["qpos"][r], f["qvel"][r])
        ours = np.asarray(env.render())
        ds = f["pixels"][r]
        d = np.abs(ours.astype(np.int16) - ds.astype(np.int16))
        row = {"row": int(r), "mean_abs": float(d.mean()), "frac_gt8": float((d > 8).mean()),
               "mean_rgb_dataset": ds.reshape(-1, 3).mean(0).round(2).tolist(),
               "mean_rgb_ours": ours.reshape(-1, 3).mean(0).round(2).tolist()}
        report["rows"].append(row)
        print(row)
        panel = np.concatenate([ds, ours, np.clip(d * 4, 0, 255).astype(np.uint8)], axis=1)
        Image.fromarray(panel).save(out / f"{name}{tag}_{r}.png")
    report["mean_abs"] = float(np.mean([x["mean_abs"] for x in report["rows"]]))
    (out / f"{name}{tag}.json").write_text(json.dumps(report, indent=2))
    print("mean_abs", report["mean_abs"], report["versions"])


if __name__ == "__main__":
    main()
