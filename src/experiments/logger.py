import csv
import json
from pathlib import Path


class ExperimentLogger:
    def __init__(self, output_root, experiment_name, scheduler_name):
        self.base_dir = Path(output_root) / experiment_name / scheduler_name
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def seed_dir(self, seed):
        path = self.base_dir / f"seed_{seed}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_seed_result(self, seed, config, result):
        output_dir = self.seed_dir(seed)
        self._write_json(output_dir / "config.json", config)
        self._write_json(output_dir / "summary.json", result["summary"])
        self._write_csv(output_dir / "workflow_results.csv", result["workflow_results"])
        self._write_csv(output_dir / "event_log.csv", result["event_log"])
        return output_dir

    def save_aggregate_summary(self, rows):
        path = self.base_dir / "aggregate_summary.csv"
        self._write_csv(path, rows)
        return path

    def _write_json(self, path, data):
        with Path(path).open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def _write_csv(self, path, rows):
        rows = rows or []
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = []
        with Path(path).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(rows)
