"""Read-only CSV sources and palette for the 2026-10-27 manuscript refresh."""
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PALETTE = {"dqn_edge": "#D99A5B", "dream_dqn": "#83A783",
           "oo_actor_critic": "#A28AA7", "d3qn_sac": "#4C78A8"}
SCALES = ("small", "medium", "large")
DEADLINES = ("loose", "medium", "tight")
RESOURCES = ("ss_37", "sm_74", "sl_111")
MODES = ("d3qn_sac", "dqn_edge", "dream_dqn", "oo_actor_critic")
CURRENT = ROOT / 'results/paper_current'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    """Preserve strings while restoring CSV scalar types; cross-check raw JSONL."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    for row in records:
        for key, value in row.items():
            if value == "":
                row[key] = None
            elif value in ("True", "False"):
                row[key] = value == "True"
            else:
                try:
                    row[key] = json.loads(value)
                except (ValueError, TypeError):
                    pass
        assert math.isfinite(row["total_energy"]) and row["total_energy"] > 0
        assert 0 <= row["deadline_success_rate"] <= 1
        assert 0 <= row["reliability_success_rate"] <= 1
    jsonl = path.with_suffix(".jsonl")
    if jsonl.exists():
        raw = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(raw) == len(records), path
        for actual, other in zip(records, raw):
            for field in ("mode", "scale", "deadline_type", "eval_seed", "num_workflows",
                          "total_energy", "deadline_success_rate", "reliability_success_rate"):
                if field in actual:
                    assert actual[field] == other[field], (path, field)
    return records


def stats(rows):
    energy = [row["total_energy"] for row in rows]
    assert len(energy) == 20
    return dict(n=20, energy_mean_J=statistics.mean(energy),
                energy_sample_sd_J=statistics.stdev(energy),
                esp_at_90=sum(r["deadline_success_rate"] >= .9-1e-9 for r in rows)/20,
                reliability_attainment=sum(r["reliability_success_rate"] >= 1-1e-9 for r in rows)/20)


def current_dataset(experiment, mode, resource=None):
    """Resolve relocated paper records through the hash-verified current manifest."""
    manifest = json.loads((CURRENT/'manifest.json').read_text(encoding='utf-8'))
    selected = [d for d in manifest['datasets'] if d['experiment']==experiment
                and d['mode']==mode and (resource is None or d['resource']==resource)]
    assert len(selected)==1, (experiment,mode,resource)
    entry=selected[0]
    data,sources=[],[]
    for path in sorted((CURRENT/entry['target_dir']).rglob('comparison_rows.csv')):
        for original in (path,path.with_suffix('.jsonl')):
            assert sha(original)==manifest['files'][str(original.relative_to(ROOT))]['sha256'],original
        sample=read_csv(path)
        data.extend(sample)
        sources.append(dict(path=str(path),sha256=sha(path),rows=len(sample)))
    assert len(data)==entry['records'],entry
    return data,sources


def independent_extra_rows(group):
    """Read completed supplementary EMA-reset records, never infer or resume."""
    if (CURRENT/'manifest.json').exists():
        data,sources=current_dataset(group,'dqn_edge')
        assert len(sources)==1
        assert all(r['ema_reset_before_rollout'] is True for r in data)
        assert len({r['ema_initial_sha256'] for r in data})==1
        return data,sources[0]
    from tools import eval_dqn_edge_independent_ema_extra as extra
    plan = extra.read(extra.OUT / "frozen_plan.json")
    extra.check_sources(plan)
    folder = extra.OUT / group
    completion = extra.read(folder / "completion.json")
    manifest = extra.read(folder / "experiment_manifest.json")
    data = read_csv(folder / "comparison_rows.csv")
    extra.validate(data, group, plan, complete=True)
    assert completion["records"] == completion["ema_reset_count"] == len(data)
    assert completion["source_hashes_verified"] is True
    assert manifest["model"] == plan["models"][group]
    assert manifest["plan_sha256"] == sha(extra.OUT / "frozen_plan.json")
    failures = [dict(setting=r["setting"], eval_seed=r["eval_seed"],
                     deadline_success_rate=r["deadline_success_rate"],
                     reliability_success_rate=r["reliability_success_rate"])
                for r in data if r["deadline_success_rate"] < .9-1e-9
                or r["reliability_success_rate"] < 1-1e-9]
    assert completion["failures"] == failures
    raw = (extra.OUT / (group + ".log")).read_bytes()
    log = raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
    assert "Traceback" not in log and log.rstrip().endswith(
        f"[extra-ema-complete] {group} records={len(data)}")
    assert not (folder / "failure.json").exists()
    path = folder / "comparison_rows.csv"
    return data, dict(path=str(path), sha256=sha(path), rows=len(data))


def main_panel(independent_ema=False):
    grouped, sources, cells = defaultdict(list), [], {}
    for resource in RESOURCES:
        for mode in MODES:
            base = "legacy_seed_block_0_19_20261026" if mode == "d3qn_sac" else "baselines_seed_block_0_19_20261026"
            folder = resource if mode == "d3qn_sac" else f"{resource}_{mode}"
            path = ROOT / "results" / base / folder / "comparison_rows.csv"
            if independent_ema and mode == "dqn_edge":
                path = ROOT / "results/dqn_edge_independent_ema_0_19_20261029" / resource / "comparison_rows.csv"
            if (CURRENT/'manifest.json').exists():
                _,current_sources=current_dataset('main_0_19',mode,resource)
                assert len(current_sources)==1
                path=Path(current_sources[0]['path'])
            rows = read_csv(path)
            keys = [(r["scale"], r["deadline_type"], r["eval_seed"]) for r in rows]
            expected = [(s, d, seed) for s in SCALES for d in DEADLINES for seed in range(20)]
            assert keys == expected and len(set(keys)) == 180, path
            assert all(r["num_workflows"] == 50 and r["mode"] == mode for r in rows)
            assert (path.parent / "completion.json").exists(), path
            for s in SCALES:
                for d in DEADLINES:
                    sample = [r for r in rows if r["scale"] == s and r["deadline_type"] == d]
                    key = resource[:2], s, d, mode
                    grouped[key] = [r["total_energy"] for r in sample]
                    cells[key] = stats(sample)
            sources.append(dict(path=str(path), sha256=sha(path), rows=180))
    assert len(cells) == 108 and sum(s["rows"] for s in sources) == 2160
    return grouped, sources, cells
