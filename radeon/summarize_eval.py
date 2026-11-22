"""Collect success rates and planning times from results/eval/*.log into eval_summary.json and a Markdown table.

python radeon/summarize_eval.py [results/eval]
"""

import json
import re
import sys
from pathlib import Path

PAPER = {"tworoom": 87, "pusht": 96, "reacher": 86, "cube": 74}
PAPER_RANDOM = {"tworoom": 0, "pusht": 2, "reacher": 10, "cube": 48}


def parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    rate = re.search(r"'success_rate': ([0-9.]+)", text)
    succ = re.search(r"'episode_successes': array\(\[([^\]]*)\]", text, re.S)
    plans = [dict(n_envs=int(n), total_s=float(t), per_env_s=float(p))
             for n, t, p in re.findall(r"PLAN n_envs=(\d+) total_s=([0-9.]+) per_env_s=([0-9.]+)", text)]
    return {
        "success_rate": round(float(rate.group(1)), 1) if rate else None,
        "episodes": len(re.findall(r"True|False", succ.group(1))) if succ else None,
        "cem_solve_s": [float(x) for x in re.findall(r"CEM solve time: ([0-9.]+) seconds", text)],
        "plans": plans,
    }


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/eval")
    runs = {p.stem: parse(p) for p in sorted(root.glob("*.log"))}
    (root.parent / "eval_summary.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    rows = ["| run | success | episodes | paper | CEM solve times (s) |", "| --- | --- | --- | --- | --- |"]
    for tag, r in runs.items():
        env = next((e for e in PAPER if e in tag), None)
        paper = PAPER_RANDOM.get(env) if "random" in tag else PAPER.get(env)
        rate = "—" if r["success_rate"] is None else f"{r['success_rate']:g}%"
        solves = ", ".join(f"{t:.1f}" for t in r["cem_solve_s"]) or "—"
        rows.append(f"| `{tag}` | {rate} | {r['episodes'] or '—'} | {paper if paper is not None else '—'} | {solves} |")
    (root.parent / "eval_summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
