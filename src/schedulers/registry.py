from src.schedulers.fastest_server_scheduler import FastestServerScheduler
from src.schedulers.fcfs_scheduler import FCFSScheduler
from src.schedulers.heft import HEFTScheduler
from src.schedulers.random_scheduler import RandomScheduler


SCHEDULER_REGISTRY = {
    "fcfs": FCFSScheduler,
    "heft": HEFTScheduler,
    "random": RandomScheduler,
    "fastest": FastestServerScheduler,
}


def _load_a2c_scheduler():
    from src.a2c.scheduler import A2CScheduler

    return A2CScheduler


def _load_ppo_scheduler():
    from src.ppo.scheduler import PPOScheduler

    return PPOScheduler


def _load_recr_ppo_scheduler():
    from src.recr.scheduler import RECRPPOScheduler

    return RECRPPOScheduler


def _load_hierarchical_safe_scheduler():
    from src.schedulers.hierarchical_safe_scheduler import HierarchicalSafeScheduler

    return HierarchicalSafeScheduler


def create_scheduler(name, params=None, seed=42):
    if name == "a2c":
        scheduler_cls = _load_a2c_scheduler()
        return scheduler_cls(params=params, seed=seed)
    if name == "ppo":
        scheduler_cls = _load_ppo_scheduler()
        return scheduler_cls(params=params, seed=seed)
    if name == "recr_ppo":
        scheduler_cls = _load_recr_ppo_scheduler()
        return scheduler_cls(params=params, seed=seed)
    if name == "hierarchical_safe_ppo":
        scheduler_cls = _load_hierarchical_safe_scheduler()
        return scheduler_cls(params=params, seed=seed)

    if name not in SCHEDULER_REGISTRY:
        available = ", ".join(
            sorted(
                [
                    *SCHEDULER_REGISTRY.keys(),
                    "a2c",
                    "ppo",
                    "recr_ppo",
                    "hierarchical_safe_ppo",
                ]
            )
        )
        raise ValueError(f"Unknown scheduler '{name}'. Available schedulers: {available}")
    return SCHEDULER_REGISTRY[name](params=params, seed=seed)
