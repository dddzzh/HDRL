from __future__ import annotations

import math
import random
from typing import Any, Mapping, Optional

# Edge-cloud DAG communication links used by the scheduling model (Mbps).
RECR_NOMINAL_LINKS_MBPS: dict[str, float] = {
    "edge_cloud": 100.0,
    "edge_edge": 1000.0,
    "cloud_cloud": 10000.0,
}

REFERENCE_SHANNON_EDGE_DEVICE = {
    "bandwidth_hz": 1.0e8,
    "transmit_power_w": 1.0,
    "noise_power_w": 1.0e-13,
    "interference_power_w": 1.0e-11,
    "path_loss_exponent": 4.0,
    "distance_m_min": 50.0,
    "distance_m_max": 100.0,
}


def resolve_link_key(src_layer: str, dst_layer: str) -> str:
    src = str(src_layer).lower()
    dst = str(dst_layer).lower()
    if src == dst:
        if src in ("edge", "cloud"):
            return f"{src}_{src}"
        raise ValueError(f"Invalid same-layer link: {src}->{dst}")
    pair = {src, dst}
    if pair == {"iot", "edge"}:
        return "iot_edge" if src == "iot" else "edge_iot"
    if pair == {"iot", "cloud"}:
        return "iot_cloud" if src == "iot" else "cloud_iot"
    if pair == {"edge", "cloud"}:
        return "edge_cloud"
    raise ValueError(f"Unsupported link layers: {src}->{dst}")


def _sample_bandwidth(rng: random.Random, nominal: float, heterogeneity: float) -> float:
    heterogeneity = min(1.0, max(0.0, float(heterogeneity)))
    low = nominal * (1.0 - heterogeneity / 2.0)
    high = nominal * (1.0 + heterogeneity / 2.0)
    return rng.uniform(low, high)


def build_recr_link_bandwidths(
    rng: random.Random,
    links_mbps: Optional[Mapping[str, float]] = None,
    heterogeneity: float = 0.0,
) -> dict[str, float]:
    nominal = dict(RECR_NOMINAL_LINKS_MBPS)
    if links_mbps:
        nominal.update({str(key): float(value) for key, value in links_mbps.items()})

    sampled: dict[str, float] = {}
    for key, value in nominal.items():
        sampled[key] = _sample_bandwidth(rng, value, heterogeneity)

    return sampled


def build_network_config(
    network: Mapping[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    model = str(network.get("model", "recr")).lower()
    if model == "flat":
        bandwidth = float(network["bandwidth_mbps"])
        return {
            "model": "flat",
            "bandwidth_mbps": bandwidth,
            "link_bandwidth_mbps": {
                "edge_cloud": bandwidth,
                "edge_edge": bandwidth,
                "cloud_cloud": bandwidth,
            },
        }

    links_mbps = network.get("links_mbps")
    heterogeneity = float(network.get("heterogeneity", 0.0))
    link_bandwidth_mbps = build_recr_link_bandwidths(
        rng=rng,
        links_mbps=links_mbps,
        heterogeneity=heterogeneity,
    )
    result = {
        "model": "recr",
        "heterogeneity": heterogeneity,
        "links_mbps": dict(RECR_NOMINAL_LINKS_MBPS if not links_mbps else links_mbps),
        "link_bandwidth_mbps": link_bandwidth_mbps,
        "bandwidth_mbps": link_bandwidth_mbps["edge_cloud"],
    }
    if "edge_device" in network:
        result["edge_device"] = dict(network["edge_device"])
    return result


def shannon_rate_mbps(
    distance_m: float,
    *,
    bandwidth_hz: float = REFERENCE_SHANNON_EDGE_DEVICE["bandwidth_hz"],
    transmit_power_w: float = REFERENCE_SHANNON_EDGE_DEVICE["transmit_power_w"],
    noise_power_w: float = REFERENCE_SHANNON_EDGE_DEVICE["noise_power_w"],
    interference_power_w: float = REFERENCE_SHANNON_EDGE_DEVICE["interference_power_w"],
    path_loss_exponent: float = REFERENCE_SHANNON_EDGE_DEVICE["path_loss_exponent"],
) -> float:
    """Reference device-edge Shannon rate, returned in Mbps."""
    distance_m = float(distance_m)
    bandwidth_hz = float(bandwidth_hz)
    transmit_power_w = float(transmit_power_w)
    noise_power_w = float(noise_power_w)
    interference_power_w = float(interference_power_w)
    path_loss_exponent = float(path_loss_exponent)
    if distance_m <= 0.0 or bandwidth_hz <= 0.0 or transmit_power_w <= 0.0:
        raise ValueError("Shannon distance, bandwidth and transmit power must be positive")
    denominator = noise_power_w + interference_power_w
    if denominator <= 0.0:
        raise ValueError("Shannon noise plus interference must be positive")
    channel_gain = distance_m ** (-path_loss_exponent)
    rate_bps = bandwidth_hz * math.log2(
        1.0 + transmit_power_w * channel_gain / denominator
    )
    return rate_bps / 1.0e6


def apply_edge_device_shannon(resources: dict[str, Any], rng: random.Random) -> None:
    """Assign each edge server the reference 50-100 m device link."""
    network = resources.get("network", {})
    edge_device = dict(network.get("edge_device", {}))
    if str(edge_device.get("model", "fixed")).lower() != "shannon_reference":
        return

    params = dict(REFERENCE_SHANNON_EDGE_DEVICE)
    params.update({key: value for key, value in edge_device.items() if key in params})
    distance_min = float(params.pop("distance_m_min"))
    distance_max = float(params.pop("distance_m_max"))
    if distance_min <= 0.0 or distance_max < distance_min:
        raise ValueError("Invalid Shannon edge-device distance range")

    rates = []
    for server in resources.get("servers", []):
        layer = str(server.get("layer", server.get("type", ""))).lower()
        if layer != "edge":
            continue
        distance_m = rng.uniform(distance_min, distance_max)
        rate_mbps = shannon_rate_mbps(distance_m, **params)
        server["infrastructure_bandwidth_mbps"] = float(server.get("bandwidth_mbps", 0.0))
        server["device_distance_m"] = distance_m
        server["device_bandwidth_mbps"] = rate_mbps
        # The reference HEFT port reads server.bandwidth_mbps as the mobile link.
        server["bandwidth_mbps"] = rate_mbps
        rates.append(rate_mbps)

    if not rates:
        raise ValueError("Shannon edge-device model requires at least one edge server")
    mean_rate = sum(rates) / len(rates)
    links = network.setdefault("link_bandwidth_mbps", {})
    links["iot_edge"] = mean_rate
    links["edge_iot"] = mean_rate
    edge_device["realized_mean_mbps"] = mean_rate
    network["edge_device"] = edge_device


def get_link_bandwidth(link_bandwidth_mbps: Mapping[str, float], src_layer: str, dst_layer: str) -> float:
    key = resolve_link_key(src_layer, dst_layer)
    if key not in link_bandwidth_mbps:
        reverse = resolve_link_key(dst_layer, src_layer)
        if reverse in link_bandwidth_mbps and reverse == "edge_cloud":
            return float(link_bandwidth_mbps[reverse])
        raise KeyError(f"Link bandwidth not configured: {key}")
    return float(link_bandwidth_mbps[key])
