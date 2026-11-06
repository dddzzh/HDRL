"""Validate ordered, real workflow seeds without importing policy code."""
import json
from pathlib import Path


def validate_seed_schedule(data, scales, deadline_types, num_seeds):
    if not isinstance(data, dict) or set(data) != set(scales):
        raise ValueError("Seed schedule must contain exactly the requested scales")
    for scale in scales:
        group = data[scale]
        if not isinstance(group, dict) or set(group) != set(deadline_types):
            raise ValueError("Seed schedule must contain exactly the requested deadlines")
        for deadline in deadline_types:
            seeds = group[deadline]
            if (not isinstance(seeds, list) or len(seeds) != num_seeds
                    or any(type(seed) is not int or seed < 0 for seed in seeds)
                    or len(set(seeds)) != num_seeds):
                raise ValueError("Each case needs the requested number of unique integer seeds")
    return data


def resolve_seed_schedule(path, scales, deadline_types, seed_start, num_seeds):
    data = (json.loads(Path(path).read_text(encoding="utf-8")) if path else
            {s: {d: list(range(seed_start, seed_start + num_seeds))
                 for d in deadline_types} for s in scales})
    return validate_seed_schedule(data, scales, deadline_types, num_seeds)
