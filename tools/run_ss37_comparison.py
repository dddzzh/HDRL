"""Train one SS37 comparator and evaluate its Small-validation checkpoint in nine scenarios."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("algorithm", choices=("dqn", "mahdrl"))
    cli.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    cli.add_argument("--eval-only", action="store_true", help="Evaluate an existing completed training run.")
    args = cli.parse_args()
    mahdrl = args.algorithm == "mahdrl"
    name = ("legacy_ss37_mahdrl_tl" if mahdrl else "dqn_edge_ss37") + "_fifs_small20_arr003_1500"
    script = "train_mahdrl_tl_legacy_ss37.py" if mahdrl else "train_dqn_edge_fair_ss37.py"
    command = [sys.executable, str(ROOT / "tools" / script)]
    train_dir = RESULTS / name
    if not args.eval_only:
        flags = (["--validate-every", "20", "--validation-count", "20", "--validation-workers", "8"]
                 if mahdrl else ["--completed-task-context", "--match-upper-state", "--energy-kj"])
        subprocess.run(command + ["--episodes", "1500", "--small20-fifs", "--device", args.device,
                       "--out", name if mahdrl else str(train_dir)] + flags, cwd=ROOT, check=True)
    checkpoint = train_dir / "best_esp90.pt"
    if not (train_dir / "episode_1500.pt").is_file() or not checkpoint.is_file():
        cli.error(f"Expected completed training and a validation-selected checkpoint in {train_dir}")
    report = []
    for scale in ("small", "middle", "large"):
        for deadline in ("loose", "medium", "tight"):
            test_dir = RESULTS / f"{name}_test_{scale}_{deadline}"
            output = test_dir / ("summary.json" if mahdrl else "evaluation.csv")
            if not output.is_file():
                flags = ["--validation-workers", "8"] if mahdrl else []
                subprocess.run(command + ["--evaluate", str(checkpoint), "--device", "cpu", "--scale", scale,
                               "--deadline-type", deadline, "--eval-workflows", "50", "--seed-start", "742000",
                               "--num-seeds", "20", "--out", test_dir.name if mahdrl else str(test_dir)] + flags,
                               cwd=ROOT, check=True)
            if mahdrl:
                rows = json.loads(output.read_text(encoding="utf-8"))["rows"]
            else:
                with output.open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
            if [int(r["seed"]) for r in rows] != list(range(742000, 742020)) or any(
                    int(r["workflow_count"]) != 50 for r in rows):
                raise ValueError(f"Incomplete test panel in {output}")
            misses = [int(r["deadline_missed_workflows"]) for r in rows]
            rel = [int(r["reliability_missed_workflows"]) for r in rows]
            report.append(dict(algorithm=args.algorithm, scale=scale, deadline=deadline, environments=20,
                               workflows=1000, ESP90=sum(d <= 5 for d in misses),
                               joint=sum(d <= 5 and r == 0 for d, r in zip(misses, rel)),
                               DDL_MISS=sum(misses), REL_MISS=sum(rel),
                               mean_energy_kJ=sum(float(r["energy_J"]) for r in rows) / len(rows) / 1000))
    path = train_dir / "nine_scenarios.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=report[0])
        writer.writeheader()
        writer.writerows(report)
    print(path)


if __name__ == "__main__":
    main()
