import copy
import json
from pathlib import Path

from src.datasets.workflow_factory import create_workflows
from src.experiments.logger import ExperimentLogger
from src.schedulers.registry import create_scheduler
from src.simulator.edge_cloud_env import EdgeCloudWorkflowEnv
from src.simulator.resource import load_resources


class ExperimentRunner:
    def __init__(self, config_path, project_root=None, overrides=None):
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
        self.config_path = self._resolve_path(config_path)
        self.overrides = overrides or {}
        self.config = self._load_config()
        self._apply_overrides()
        self._normalize_paths()

    def run_all(self):
        seeds = self.config.get("runs", {}).get("seeds", [42])
        scheduler_name = self.config["scheduler"]["name"]
        logger = ExperimentLogger(
            output_root=self.config.get("runs", {}).get("output_root", "results"),
            experiment_name=self.config["experiment_name"],
            scheduler_name=scheduler_name,
        )

        aggregate_rows = []
        seed_outputs = []
        for seed in seeds:
            run_config = copy.deepcopy(self.config)
            resource_config = run_config["resources"]
            resources = load_resources(
                resource_config["path"],
                num_edge=resource_config.get("num_edge"),
                num_cloud=resource_config.get("num_cloud"),
                seed=seed,
            )
            workflows = self._load_dataset(run_config["dataset"], seed, resources=resources)
            run_config["resources"]["instances"] = copy.deepcopy(resources["servers"])
            scheduler = create_scheduler(
                scheduler_name,
                params=run_config.get("scheduler", {}).get("params", {}),
                seed=seed,
            )
            env = EdgeCloudWorkflowEnv(
                workflows=workflows,
                resources=resources,
                sim_config=run_config.get("simulation", {}),
                seed=seed,
            )
            result = env.run(scheduler)
            output_dir = logger.save_seed_result(seed, run_config, result)
            seed_outputs.append((seed, output_dir, result))
            aggregate_rows.append(self._aggregate_row(seed, result["summary"]))

        aggregate_path = logger.save_aggregate_summary(aggregate_rows)
        return {
            "config": self.config,
            "seed_outputs": seed_outputs,
            "aggregate_path": aggregate_path,
            "aggregate_rows": aggregate_rows,
        }

    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        self._validate_config(config)
        return config

    def _apply_overrides(self):
        if self.overrides.get("scheduler"):
            self.config["scheduler"]["name"] = self.overrides["scheduler"]
        if self.overrides.get("dataset"):
            self.config["dataset"]["path"] = self.overrides["dataset"]
        if self.overrides.get("seeds") is not None:
            self.config["runs"]["seeds"] = self.overrides["seeds"]
        if self.overrides.get("num_edge") is not None:
            self.config["resources"]["num_edge"] = int(self.overrides["num_edge"])
        if self.overrides.get("num_cloud") is not None:
            self.config["resources"]["num_cloud"] = int(self.overrides["num_cloud"])

    def _normalize_paths(self):
        self.config["dataset"]["path"] = str(self._resolve_path(self.config["dataset"]["path"]))
        self.config["resources"]["path"] = str(self._resolve_path(self.config["resources"]["path"]))
        self.config["runs"]["output_root"] = str(
            self._resolve_path(self.config.get("runs", {}).get("output_root", "results"))
        )

    def _resolve_path(self, path):
        resolved = Path(path)
        if resolved.is_absolute():
            return resolved
        return self.project_root / resolved

    def _validate_config(self, config):
        for field in ("experiment_name", "dataset", "resources", "scheduler", "simulation", "runs"):
            if field not in config:
                raise ValueError(f"Experiment config missing required field '{field}'")
        if "path" not in config["dataset"]:
            raise ValueError("Experiment config dataset missing required field 'path'")
        if "path" not in config["resources"]:
            raise ValueError("Experiment config resources missing required field 'path'")
        resources = config["resources"]
        if resources.get("num_edge") is None or resources.get("num_cloud") is None:
            raise ValueError(
                "Experiment config resources missing required fields 'num_edge' and 'num_cloud'"
            )
        if "name" not in config["scheduler"]:
            raise ValueError("Experiment config scheduler missing required field 'name'")

    def _load_dataset(self, dataset_config, seed, resources=None):
        dataset_type = dataset_config.get("type")
        if dataset_type == "alibaba_json":
            return create_workflows(dataset_config, source="alibaba_json", seed=seed, resources=resources)
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    def _aggregate_row(self, seed, summary):
        row = {"seed": seed}
        row.update(summary)
        return row
