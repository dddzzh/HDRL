from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "configs" / "resources_edge_cloud_default.json"


def build_resource_template(
    bandwidth_mbps: float = 100.0,
) -> dict[str, Any]:
    return {
        "frequency_model": "normalized_dvfs",
        "rho_levels": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        "network": {
            "bandwidth_mbps": bandwidth_mbps,
        },
        "server_types": {
            "edge": {
                "max_freq_ghz": [2.0, 3.0],
                "static_power": [6.0, 13.0],
                "dynamic_power": [14.0, 42.0],
                "memory_gb": [4.0, 16.0],
                "failure_rate": [1.0e-4, 1.5e-4],
                "failure_rate_exponent": 1.0,
            },
            "cloud": {
                "max_freq_ghz": [4.0, 5.0],
                "static_power": [70.0, 100.0],
                "dynamic_power": [90.0, 160.0],
                "memory_gb": [32.0, 128.0],
                "failure_rate": [5.0e-5, 1.0e-4],
                "failure_rate_exponent": 1.0,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write edge/cloud resource parameter template (ranges only, no fixed servers)."
    )
    parser.add_argument(
        "--bandwidth-mbps",
        type=float,
        default=100.0,
        help="Uniform inter-server bandwidth in Mbps",
    )
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    payload = build_resource_template(
        bandwidth_mbps=args.bandwidth_mbps,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Wrote resource template to {output_path}")
    print("Set num_edge and num_cloud in experiment config when creating the environment.")


if __name__ == "__main__":
    main()
