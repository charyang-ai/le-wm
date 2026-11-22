"""Run le-wm eval.py with every CEM solve timed per environment.

WorldModelPolicy only replans environments that are still running, so the "CEM solve time" eval.py prints
covers a different number of environments each call. This prints one PLAN line per solve with that count.
    python radeon/plan_timing.py --config-name=pusht.yaml policy=quentinll/lewm-pusht
"""

import runpy
import sys
import time

import torch
from stable_worldmodel.solver.cem import CEMSolver

_solve = CEMSolver.solve


def timed_solve(self, info_dict, init_action=None):
    n = len(next(iter(info_dict.values())))
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = _solve(self, info_dict, init_action)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    print(f"PLAN n_envs={n} total_s={dt:.4f} per_env_s={dt / n:.4f}", flush=True)
    return out


CEMSolver.solve = timed_solve
sys.argv = ["eval.py", *sys.argv[1:]]
runpy.run_path("eval.py", run_name="__main__")
