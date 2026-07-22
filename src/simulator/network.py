from __future__ import annotations

import random
from typing import Any, Mapping, Optional

# RECR Section 5.2 nominal link bandwidths (Mb/s, treated as Mbps).
RECR_NOMINAL_LINKS_MBPS: dict[str, float] = {
    "iot_edge": 50.0,
    "iot_cloud": 20.0,
    "edge_cloud": 100.0,
    "edge_edge": 1000.0,
    "cloud_cloud": 10000.0,
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

    # Egress uses the same nominal values as ingress (symmetric IoT links).
    sampled["edge_iot"] = sampled["iot_edge"]
    sampled["cloud_iot"] = sampled["iot_cloud"]
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
                "iot_edge": bandwidth,
                "iot_cloud": bandwidth,
                "edge_iot": bandwidth,
                "cloud_iot": bandwidth,
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
    return {
        "model": "recr",
        "heterogeneity": heterogeneity,
        "links_mbps": dict(RECR_NOMINAL_LINKS_MBPS if not links_mbps else links_mbps),
        "link_bandwidth_mbps": link_bandwidth_mbps,
        "bandwidth_mbps": link_bandwidth_mbps["edge_cloud"],
    }


def get_link_bandwidth(link_bandwidth_mbps: Mapping[str, float], src_layer: str, dst_layer: str) -> float:
    key = resolve_link_key(src_layer, dst_layer)
    if key not in link_bandwidth_mbps:
        reverse = resolve_link_key(dst_layer, src_layer)
        if reverse in link_bandwidth_mbps and reverse in ("edge_iot", "cloud_iot", "edge_cloud"):
            return float(link_bandwidth_mbps[reverse])
        raise KeyError(f"Link bandwidth not configured: {key}")
    return float(link_bandwidth_mbps[key])
