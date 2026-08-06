import copy
import json
import random
from pathlib import Path

from src.simulator.network import apply_edge_device_shannon, build_network_config


SERVER_RANGE_FIELDS = (
    "max_freq_ghz",
    "static_power",
    "dynamic_power",
    "memory_gb",
    "failure_rate",
)

PROFILE_FIELD_NAMES = (
    "max_freq_ghz",
    "static_power",
    "dynamic_power",
    "active_power_reference",
    "memory_gb",
    "bandwidth_mbps",
    "failure_rate",
    "min_rho",
)

SERVER_PROFILE_ORDER = (
    "cloud_fast",
    "cloud_standard",
    "edge_efficient",
    "edge_legacy",
    "edge_lowend",
)

SERVER_PROFILE_GENERATION_ORDER = (
    "edge_efficient",
    "edge_legacy",
    "edge_lowend",
    "cloud_fast",
    "cloud_standard",
)


def load_resources(path, num_edge=None, num_cloud=None, seed=42):
    resource_path = Path(path)
    if not resource_path.exists():
        raise FileNotFoundError(f"Resource config not found: {resource_path}")

    with resource_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    resources = copy.deepcopy(config)
    network = resources.get("network")
    if not isinstance(network, dict):
        raise ValueError(f"Resource config missing 'network' object: {resource_path}")
    if "model" not in network and "bandwidth_mbps" not in network:
        raise ValueError(
            f"Resource config network must define 'model' (recr) or legacy 'bandwidth_mbps': "
            f"{resource_path}"
        )

    rng = random.Random(seed)
    if network.get("model", "recr") == "flat" or (
        "model" not in network and "bandwidth_mbps" in network
    ):
        flat_network = dict(network)
        flat_network.setdefault("model", "flat")
        resources["network"] = build_network_config(flat_network, rng)
    else:
        resources["network"] = build_network_config(network, rng)

    if "server_types" in resources:
        if num_edge is None or num_cloud is None:
            raise ValueError(
                "Resource template requires num_edge and num_cloud when creating the environment"
            )
        if int(num_edge) < 0 or int(num_cloud) < 0:
            raise ValueError("num_edge and num_cloud must be non-negative")
        if int(num_edge) + int(num_cloud) == 0:
            raise ValueError("At least one server is required")

        resources["servers"] = generate_servers(
            server_types=resources["server_types"],
            num_edge=int(num_edge),
            num_cloud=int(num_cloud),
            rng=rng,
            debug_neutral_reliability=bool(resources.get("debug_neutral_reliability", False)),
        )
        resources["num_edge"] = int(num_edge)
        resources["num_cloud"] = int(num_cloud)
    elif "servers" not in resources or not isinstance(resources["servers"], list):
        raise ValueError(
            f"Resource config must define 'server_types' or list field 'servers': {resource_path}"
        )

    for server in resources["servers"]:
        _normalize_server(server)

    apply_edge_device_shannon(resources, rng)

    print_resource_diagnostics(resources)
    return resources


def generate_servers(server_types, num_edge, num_cloud, rng, debug_neutral_reliability=False):
    if _has_profile_server_types(server_types):
        return _generate_profile_servers(
            server_types=server_types,
            num_edge=num_edge,
            num_cloud=num_cloud,
            rng=rng,
            debug_neutral_reliability=debug_neutral_reliability,
        )

    if "edge" not in server_types or "cloud" not in server_types:
        raise ValueError(
            "Resource template server_types must include edge/cloud ranges or the five profile keys: "
            + ", ".join(SERVER_PROFILE_ORDER)
        )

    servers = []
    server_id = 0
    for local_index in range(num_edge):
        servers.append(
            _sample_server(
                server_id=server_id,
                server_type="edge",
                local_index=local_index,
                spec=server_types["edge"],
                rng=rng,
            )
        )
        server_id += 1

    for local_index in range(num_cloud):
        servers.append(
            _sample_server(
                server_id=server_id,
                server_type="cloud",
                local_index=local_index,
                spec=server_types["cloud"],
                rng=rng,
            )
        )
        server_id += 1
    return servers


def _has_profile_server_types(server_types):
    return all(name in server_types for name in SERVER_PROFILE_ORDER)


def _generate_profile_servers(
    server_types,
    num_edge,
    num_cloud,
    rng,
    debug_neutral_reliability=False,
):
    counts = _profile_counts(num_edge=num_edge, num_cloud=num_cloud)
    servers = []
    server_id = 0
    for profile_name in SERVER_PROFILE_GENERATION_ORDER:
        spec = server_types[profile_name]
        for local_index in range(counts[profile_name]):
            servers.append(
                _sample_profile_server(
                    server_id=server_id,
                    profile_name=profile_name,
                    local_index=local_index,
                    spec=spec,
                    rng=rng,
                    debug_neutral_reliability=debug_neutral_reliability,
                )
            )
            server_id += 1
    return servers


def _profile_counts(num_edge, num_cloud):
    cloud_fast = (int(num_cloud) + 1) // 2
    cloud_standard = int(num_cloud) // 2
    edge_base = int(num_edge) // 3
    edge_remainder = int(num_edge) % 3
    return {
        "cloud_fast": cloud_fast,
        "cloud_standard": cloud_standard,
        "edge_efficient": edge_base + (1 if edge_remainder > 0 else 0),
        "edge_legacy": edge_base + (1 if edge_remainder > 1 else 0),
        "edge_lowend": edge_base,
    }


def _sample_profile_server(
    server_id,
    profile_name,
    local_index,
    spec,
    rng,
    debug_neutral_reliability=False,
):
    for field in PROFILE_FIELD_NAMES:
        if field not in spec:
            raise ValueError(f"Server profile '{profile_name}' missing field '{field}'")

    layer = str(spec["layer"])
    max_freq_ghz = float(spec["max_freq_ghz"]) * rng.uniform(0.95, 1.05)
    static_power = float(spec["static_power"]) * rng.uniform(0.95, 1.05)
    dynamic_power = float(spec["dynamic_power"]) * rng.uniform(0.95, 1.05)
    bandwidth_mbps = float(spec["bandwidth_mbps"]) * rng.uniform(0.9, 1.1)
    failure_rate = 1.0e-4 if debug_neutral_reliability else float(spec["failure_rate"])

    server = {
        "id": server_id,
        "server_id": server_id,
        "name": f"{profile_name}_{local_index}",
        "type": layer,
        "server_type": profile_name,
        "layer": layer,
        "max_freq_ghz": max_freq_ghz,
        "max_freq_hz": max_freq_ghz * 1e9,
        "static_power": static_power,
        "dynamic_power": dynamic_power,
        "active_power_reference": float(spec["active_power_reference"]),
        "memory_gb": float(spec["memory_gb"]),
        "bandwidth_mbps": bandwidth_mbps,
        "failure_rate": failure_rate,
        "min_rho": float(spec["min_rho"]),
    }
    if "failure_rate_exponent" in spec:
        server["failure_rate_exponent"] = float(spec["failure_rate_exponent"])
    return server


def _sample_server(server_id, server_type, local_index, spec, rng):
    for field in SERVER_RANGE_FIELDS:
        if field not in spec:
            raise ValueError(f"Server type '{server_type}' missing range field '{field}'")

    max_freq_ghz = _sample_range(rng, spec["max_freq_ghz"])
    server = {
        "id": server_id,
        "server_id": server_id,
        "name": f"{server_type}_{local_index}",
        "type": server_type,
        "server_type": server_type,
        "layer": server_type,
        "max_freq_ghz": max_freq_ghz,
        "max_freq_hz": max_freq_ghz * 1e9,
        "static_power": _sample_range(rng, spec["static_power"]),
        "dynamic_power": _sample_range(rng, spec["dynamic_power"]),
        "memory_gb": _sample_range(rng, spec["memory_gb"]),
        "bandwidth_mbps": _sample_range(rng, spec.get("bandwidth_mbps", [0.0, 0.0])),
        "failure_rate": _sample_range(rng, spec["failure_rate"]),
        "min_rho": float(spec.get("min_rho", 0.1)),
    }
    if "active_power_reference" in spec:
        server["active_power_reference"] = float(spec["active_power_reference"])
    if "failure_rate_exponent" in spec:
        server["failure_rate_exponent"] = float(spec["failure_rate_exponent"])
    return server


def _sample_range(rng, range_pair):
    low = float(range_pair[0])
    high = float(range_pair[1])
    if low > high:
        raise ValueError(f"Invalid range [{low}, {high}]: min must be <= max")
    return rng.uniform(low, high)


def _normalize_server(server):
    for field in ("id", "name", "type", "static_power", "dynamic_power"):
        if field not in server:
            raise ValueError(f"Server resource missing required field '{field}'")
    # primary capability is frequency in Hz; derive it from GHz when missing.
    if "max_freq_hz" not in server:
        if "max_freq_ghz" in server:
            server["max_freq_hz"] = float(server["max_freq_ghz"]) * 1e9
        else:
            raise ValueError(
                f"Server {server.get('id')} missing 'max_freq_hz' / 'max_freq_ghz'"
            )
    if "max_freq_ghz" not in server:
        server["max_freq_ghz"] = float(server["max_freq_hz"]) / 1e9
    server["id"] = int(server["id"])
    server["server_id"] = int(server.get("server_id", server["id"]))
    server["server_type"] = str(server.get("server_type", server.get("type", "")))
    server["layer"] = str(server.get("layer", server["type"]))
    server["type"] = str(server.get("type", server["layer"]))
    server["max_freq_ghz"] = float(server["max_freq_ghz"])
    server["max_freq_hz"] = float(server["max_freq_hz"])
    server["static_power"] = float(server["static_power"])
    server["dynamic_power"] = float(server["dynamic_power"])
    server["active_power_reference"] = float(
        server.get("active_power_reference", server["static_power"] + server["dynamic_power"])
    )
    server["memory_gb"] = float(server.get("memory_gb", 0.0))
    server["bandwidth_mbps"] = float(server.get("bandwidth_mbps", 0.0))
    server["failure_rate"] = float(server.get("failure_rate", 0.0))
    if "failure_rate_exponent" in server:
        server["failure_rate_exponent"] = float(server["failure_rate_exponent"])
    server["min_rho"] = float(server.get("min_rho", 0.1))
    server["min_freq_hz"] = float(server.get("min_freq_hz", server["min_rho"] * server["max_freq_hz"]))
    server["execution_available_time"] = 0.0
    server["backup_available_time"] = 0.0
    # Backward-compatible alias for schedulers that still read a single queue.
    server["available_time"] = server["execution_available_time"]


def clone_runtime_resources(resources):
    runtime = copy.deepcopy(resources)
    for server in runtime["servers"]:
        server["execution_available_time"] = 0.0
        server["backup_available_time"] = 0.0
        server["available_time"] = server["execution_available_time"]
    return runtime


def get_actual_freq_hz(server, rho):
    """Effective single-core frequency under DVFS, in cycles/s."""
    return float(rho) * float(server["max_freq_hz"])


def compute_execution_time(workload_cycles, server, rho):
    """Execution time in seconds: T = L / (rho * f), L in cycles, f in Hz."""
    actual_freq_hz = get_actual_freq_hz(server, rho)
    if actual_freq_hz <= 0:
        raise ValueError(
            f"Invalid actual_freq_hz for server {server.get('id')}: {actual_freq_hz}"
        )
    return float(workload_cycles) / actual_freq_hz


def compute_power(server, rho):
    """Power in W: P = P_static + P_dynamic_max * rho^3.

    ``dynamic_power`` is the dynamic power at rho=1.0, already expressed in W.
    It represents the alpha*C*V^2*f term at maximum voltage/frequency, so DVFS
    scales it by rho^3 when V and f are reduced together.
    """
    rho = float(rho)
    static_power = float(server["static_power"])
    dynamic_power = float(server["dynamic_power"]) * (rho ** 3)
    return static_power + dynamic_power


def compute_energy(workload_cycles, server, rho):
    exec_time = compute_execution_time(workload_cycles, server, rho)
    power = compute_power(server, rho)
    return power * exec_time


def print_resource_diagnostics(resources):
    servers = list(resources.get("servers", []))
    if not servers:
        return
    print("[resources] server profiles")
    counts = {}
    by_type = {}
    for server in servers:
        server_type = str(server.get("server_type", server.get("type", "unknown")))
        counts[server_type] = counts.get(server_type, 0) + 1
        by_type.setdefault(server_type, []).append(server)
        print(
            "  "
            f"server_id={server.get('server_id', server.get('id'))} "
            f"server_type={server_type} "
            f"layer={server.get('layer', server.get('type'))} "
            f"max_freq_ghz={float(server.get('max_freq_ghz', 0.0)):.4f} "
            f"static_power={float(server.get('static_power', 0.0)):.4f} "
            f"dynamic_power={float(server.get('dynamic_power', 0.0)):.4f} "
            f"bandwidth_mbps={float(server.get('bandwidth_mbps', 0.0)):.2f}"
            + (
                f" device_distance_m={float(server['device_distance_m']):.2f}"
                if server.get("device_distance_m") is not None
                else ""
            )
        )
    count_text = "; ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    print(f"[resources] profile_counts {count_text}")
    for server_type, rows in sorted(by_type.items()):
        avg_freq = sum(float(row.get("max_freq_ghz", 0.0)) for row in rows) / max(len(rows), 1)
        avg_power = sum(
            float(row.get("static_power", 0.0)) + float(row.get("dynamic_power", 0.0))
            for row in rows
        ) / max(len(rows), 1)
        print(
            f"[resources] profile_avg server_type={server_type} "
            f"count={len(rows)} avg_max_freq_ghz={avg_freq:.4f} avg_power={avg_power:.4f}W"
        )
    _warn_dominant_servers(servers)


def _warn_dominant_servers(servers):
    threshold = max(3, len(servers) // 4)
    for server in servers:
        dominated = []
        for other in servers:
            if int(server.get("id")) == int(other.get("id")):
                continue
            better_or_equal = (
                float(server.get("max_freq_ghz", 0.0)) >= float(other.get("max_freq_ghz", 0.0))
                and _total_power(server) <= _total_power(other)
                and float(server.get("failure_rate", 0.0)) <= float(other.get("failure_rate", 0.0))
                and float(server.get("bandwidth_mbps", 0.0)) >= float(other.get("bandwidth_mbps", 0.0))
            )
            strictly_better = (
                float(server.get("max_freq_ghz", 0.0)) > float(other.get("max_freq_ghz", 0.0))
                or _total_power(server) < _total_power(other)
                or float(server.get("failure_rate", 0.0)) < float(other.get("failure_rate", 0.0))
                or float(server.get("bandwidth_mbps", 0.0)) > float(other.get("bandwidth_mbps", 0.0))
            )
            if better_or_equal and strictly_better:
                dominated.append(other)
        if len(dominated) >= threshold:
            print(
                "[resources][warning] "
                f"server_id={server.get('id')} server_type={server.get('server_type')} "
                f"dominates {len(dominated)}/{len(servers) - 1} servers "
                "(max_freq_ghz higher/equal, power lower/equal, failure_rate lower/equal, bandwidth higher/equal)"
            )


def _total_power(server):
    return float(server.get("static_power", 0.0)) + float(server.get("dynamic_power", 0.0))


def compute_transfer_time(data_size_mb, bandwidth_mbps):
    data_size_mb = float(data_size_mb)
    bandwidth_mbps = float(bandwidth_mbps)
    if data_size_mb <= 0:
        return 0.0
    if bandwidth_mbps <= 0:
        raise ValueError(f"Invalid bandwidth_mbps: {bandwidth_mbps}")
    data_size_mbits = data_size_mb * 8.0
    return data_size_mbits / bandwidth_mbps
