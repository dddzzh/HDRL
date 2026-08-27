from __future__ import annotations

import copy

import numpy as np

from src.a2c.a2c import _require_torch
from src.device_utils import resolve_agent_device
from src.ppo.hierarchical_agent import _assert_finite_tensor, _build_mlp

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = object
    F = None


EPS = 1e-8


class CoordinationReplayBuffer:
    """Replay for the observed discrete-server/continuous-DVFS joint action."""

    def __init__(self, state_dim, action_dim, max_size=50000, batch_size=256):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_size = int(max_size)
        self.batch_size = int(batch_size)
        self.mem_count = 0
        self.states = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.server_actions = np.zeros(self.max_size, dtype=np.int64)
        self.rhos = np.zeros(self.max_size, dtype=np.float32)
        self.energy_costs = np.zeros(self.max_size, dtype=np.float32)
        self.deadline_costs = np.zeros(self.max_size, dtype=np.float32)
        self.next_states = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.next_server_actions = np.zeros(self.max_size, dtype=np.int64)
        self.next_rhos = np.ones(self.max_size, dtype=np.float32)
        self.dones = np.zeros(self.max_size, dtype=bool)

    @property
    def size(self):
        return min(self.mem_count, self.max_size)

    def ready(self):
        return self.size >= self.batch_size

    def store(
        self,
        state,
        server_action,
        rho,
        energy_cost,
        deadline_cost,
        next_state,
        next_server_action,
        next_rho,
        done,
    ):
        index = self.mem_count % self.max_size
        server_action = int(server_action)
        next_server_action = int(next_server_action)
        values = np.asarray(
            [rho, energy_cost, deadline_cost, next_rho], dtype=np.float64
        )
        if not 0 <= server_action < self.action_dim:
            raise ValueError("Coordination server action is out of range.")
        if not 0 <= next_server_action < self.action_dim:
            raise ValueError("Coordination next server action is out of range.")
        if not np.isfinite(values).all():
            raise FloatingPointError("Coordination replay contains NaN or Inf.")
        if not 0.0 <= float(rho) <= 1.0 or not 0.0 <= float(next_rho) <= 1.0:
            raise ValueError("Coordination rho must be in [0, 1].")
        if float(energy_cost) < 0.0 or float(deadline_cost) < 0.0:
            raise ValueError("Coordination costs must be non-negative.")
        self.states[index] = np.asarray(state, dtype=np.float32)
        self.server_actions[index] = server_action
        self.rhos[index] = float(rho)
        self.energy_costs[index] = float(energy_cost)
        self.deadline_costs[index] = float(deadline_cost)
        self.next_states[index] = np.asarray(next_state, dtype=np.float32)
        self.next_server_actions[index] = next_server_action
        self.next_rhos[index] = float(next_rho)
        self.dones[index] = bool(done)
        self.mem_count += 1

    def sample(self, rng):
        if not self.ready():
            return None
        indices = rng.choice(self.size, self.batch_size, replace=False)
        return (
            self.states[indices],
            self.server_actions[indices],
            self.rhos[indices],
            self.energy_costs[indices],
            self.deadline_costs[indices],
            self.next_states[indices],
            self.next_server_actions[indices],
            self.next_rhos[indices],
            self.dones[indices],
        )

    def state_dict(self):
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_size": self.max_size,
            "batch_size": self.batch_size,
            "mem_count": self.mem_count,
            "states": self.states,
            "server_actions": self.server_actions,
            "rhos": self.rhos,
            "energy_costs": self.energy_costs,
            "deadline_costs": self.deadline_costs,
            "next_states": self.next_states,
            "next_server_actions": self.next_server_actions,
            "next_rhos": self.next_rhos,
            "dones": self.dones,
        }

    def load_state_dict(self, payload):
        if int(payload.get("state_dim", self.state_dim)) != self.state_dim:
            raise ValueError("Coordination replay state dimension is incompatible.")
        if int(payload.get("action_dim", self.action_dim)) != self.action_dim:
            raise ValueError("Coordination replay action dimension is incompatible.")
        self.mem_count = int(payload["mem_count"])
        for name in (
            "states",
            "server_actions",
            "rhos",
            "energy_costs",
            "deadline_costs",
            "next_states",
            "next_server_actions",
            "next_rhos",
            "dones",
        ):
            getattr(self, name)[:] = payload[name]


class CoordinationCostNetwork(nn.Module):
    """Joint cost critic with separate energy and Deadline heads."""

    def __init__(self, state_dim, action_dim, hidden_dim=256, hidden_layers=2):
        _require_torch()
        super().__init__()
        input_dim = int(state_dim) + int(action_dim) + 1
        self.action_dim = int(action_dim)
        self.backbone = _build_mlp(input_dim, hidden_dim, hidden_layers)
        self.energy_head = nn.Linear(int(hidden_dim), 1)
        self.deadline_head = nn.Linear(int(hidden_dim), 1)

    def forward(self, states, server_actions, rhos):
        one_hot = F.one_hot(
            server_actions.long(), num_classes=self.action_dim
        ).to(dtype=states.dtype)
        if rhos.ndim == 1:
            rhos = rhos.unsqueeze(-1)
        features = self.backbone(torch.cat((states, one_hot, rhos), dim=-1))
        return (
            F.softplus(self.energy_head(features)).squeeze(-1),
            F.softplus(self.deadline_head(features)).squeeze(-1),
        )


class HierarchicalCoordinationCritic:
    """Conservative twin critic for discrete placement/continuous DVFS coupling.

    The critic learns two stable cost streams.  It is not inserted into either
    environment reward.  Once warm, it can re-rank D3QN candidates using the
    deterministic rho predicted by the current low SAC policy.
    """

    def __init__(self, state_dim, action_dim, config=None, seed=42):
        _require_torch()
        config = dict(config or {})
        self.config = copy.deepcopy(config)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.observed_transitions = 0
        self.learn_updates = 0
        self.device = resolve_agent_device(config.get("device", "auto"))
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 0.005))
        self.learning_rate = float(config.get("learning_rate", 1e-4))
        self.training_per_episode = int(config.get("training_per_episode", 5))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.energy_scale_joules = float(config.get("energy_scale_joules", 1000.0))
        self.deadline_cost_weight = float(config.get("deadline_cost_weight", 1.0))
        self.beta_final = float(config.get("beta_final", 0.2))
        self.warmup_transitions = int(config.get("warmup_transitions", 20000))
        self.beta_anneal_transitions = int(
            config.get("beta_anneal_transitions", 20000)
        )
        if not 0.0 <= self.beta_final <= 1.0:
            raise ValueError("coordination beta_final must be in [0, 1].")
        if self.energy_scale_joules <= 0.0:
            raise ValueError("coordination energy_scale_joules must be positive.")
        if self.deadline_cost_weight < 0.0:
            raise ValueError("coordination deadline_cost_weight must be non-negative.")
        if self.warmup_transitions < 0 or self.beta_anneal_transitions < 0:
            raise ValueError("coordination transition schedules must be non-negative.")

        hidden_dim = int(config.get("hidden_dim", 256))
        hidden_layers = int(config.get("hidden_layers", 2))
        # Initialize the auxiliary network reproducibly without changing the
        # global torch RNG stream used by the already-created D3QN and SAC.
        fork_devices = (
            list(range(torch.cuda.device_count()))
            if torch.cuda.is_available()
            else []
        )
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(self.seed)
            self.critic_1 = CoordinationCostNetwork(
                self.state_dim, self.action_dim, hidden_dim, hidden_layers
            ).to(self.device)
            self.critic_2 = CoordinationCostNetwork(
                self.state_dim, self.action_dim, hidden_dim, hidden_layers
            ).to(self.device)
            self.target_1 = CoordinationCostNetwork(
                self.state_dim, self.action_dim, hidden_dim, hidden_layers
            ).to(self.device)
            self.target_2 = CoordinationCostNetwork(
                self.state_dim, self.action_dim, hidden_dim, hidden_layers
            ).to(self.device)
        self.optimizer_1 = torch.optim.Adam(
            self.critic_1.parameters(), lr=self.learning_rate
        )
        self.optimizer_2 = torch.optim.Adam(
            self.critic_2.parameters(), lr=self.learning_rate
        )
        self.replay = CoordinationReplayBuffer(
            self.state_dim,
            self.action_dim,
            max_size=int(config.get("replay_size", 50000)),
            batch_size=int(config.get("batch_size", 256)),
        )
        self.update_targets(tau=1.0)

    @property
    def beta(self):
        excess = max(0, self.observed_transitions - self.warmup_transitions)
        if excess <= 0:
            return 0.0
        if self.beta_anneal_transitions <= 0:
            return self.beta_final
        return float(
            self.beta_final
            * min(1.0, excess / float(self.beta_anneal_transitions))
        )

    @property
    def guidance_active(self):
        return bool(self.beta > 0.0 and self.learn_updates > 0)

    def conservative_costs(self, states, server_actions, rhos):
        with torch.no_grad():
            e1, d1 = self.critic_1(states, server_actions, rhos)
            e2, d2 = self.critic_2(states, server_actions, rhos)
        return torch.maximum(e1, e2), torch.maximum(d1, d2)

    def candidate_coordination_values(
        self, state, server_actions, rhos, deadline_lambda
    ):
        states = np.repeat(
            np.asarray(state, dtype=np.float32)[None, :],
            len(server_actions),
            axis=0,
        )
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(
            server_actions, dtype=torch.long, device=self.device
        )
        rhos_t = torch.as_tensor(rhos, dtype=torch.float32, device=self.device)
        energy_cost, deadline_cost = self.conservative_costs(
            states_t, actions_t, rhos_t
        )
        values = -energy_cost - (
            float(self.deadline_cost_weight)
            * float(max(deadline_lambda, 0.0))
            * deadline_cost
        )
        return {
            "values": values.detach().cpu().numpy().astype(np.float64),
            "energy_costs": energy_cost.detach().cpu().numpy().astype(np.float64),
            "deadline_costs": deadline_cost.detach().cpu().numpy().astype(np.float64),
        }

    def remember_from_high_buffer(self, high_buffer):
        stored = 0
        if high_buffer.size <= 0:
            return stored
        zero_state = np.zeros(self.state_dim, dtype=np.float32)
        for index in range(high_buffer.size):
            info = high_buffer.infos[index]
            rho = float(info.get("coordination_low_rho", 1.0))
            energy_joules = max(float(info.get("total_energy_actual", 0.0)), 0.0)
            deadline_cost = max(
                float(info.get("high_deadline_constraint_cost", 0.0)), 0.0
            )
            done = index == high_buffer.size - 1
            if done:
                next_state = zero_state
                next_action = 0
                next_rho = 1.0
            else:
                next_state = high_buffer.states[index + 1]
                next_action = int(high_buffer.primary_actions[index + 1])
                next_rho = float(
                    high_buffer.infos[index + 1].get("coordination_low_rho", 1.0)
                )
            self.replay.store(
                high_buffer.states[index],
                int(high_buffer.primary_actions[index]),
                rho,
                energy_joules / self.energy_scale_joules,
                deadline_cost,
                next_state,
                next_action,
                next_rho,
                done,
            )
            stored += 1
        self.observed_transitions += stored
        return stored

    def update(self, high_buffer=None):
        stored = self.remember_from_high_buffer(high_buffer) if high_buffer is not None else 0
        totals = {
            "replay_size": float(self.replay.size),
            "stored_transitions": float(stored),
            "observed_transitions": float(self.observed_transitions),
            "learn_updates_total": float(self.learn_updates),
            "beta": self.beta,
            "guidance_active": float(self.guidance_active),
        }
        updates = 0
        for _ in range(self.training_per_episode):
            metrics = self.learn()
            if not metrics:
                continue
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            updates += 1
        if updates:
            for key in list(totals):
                if key in {
                    "replay_size",
                    "stored_transitions",
                    "observed_transitions",
                    "learn_updates_total",
                    "beta",
                    "guidance_active",
                }:
                    continue
                totals[key] /= updates
        totals["updates"] = float(updates)
        totals["observed_transitions"] = float(self.observed_transitions)
        totals["learn_updates_total"] = float(self.learn_updates)
        totals["beta"] = self.beta
        totals["guidance_active"] = float(self.guidance_active)
        return totals

    def learn(self):
        sample = self.replay.sample(self.rng)
        if sample is None:
            return {}
        (
            states,
            actions,
            rhos,
            energy_costs,
            deadline_costs,
            next_states,
            next_actions,
            next_rhos,
            dones,
        ) = sample
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rhos_t = torch.as_tensor(rhos, dtype=torch.float32, device=self.device)
        energy_t = torch.as_tensor(
            energy_costs, dtype=torch.float32, device=self.device
        )
        deadline_t = torch.as_tensor(
            deadline_costs, dtype=torch.float32, device=self.device
        )
        next_states_t = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        next_actions_t = torch.as_tensor(
            next_actions, dtype=torch.long, device=self.device
        )
        next_rhos_t = torch.as_tensor(
            next_rhos, dtype=torch.float32, device=self.device
        )
        dones_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            ne1, nd1 = self.target_1(
                next_states_t, next_actions_t, next_rhos_t
            )
            ne2, nd2 = self.target_2(
                next_states_t, next_actions_t, next_rhos_t
            )
            # Conservative cost-to-go: a candidate is not declared safe or
            # cheap merely because one critic is optimistic.
            next_energy = torch.maximum(ne1, ne2)
            next_deadline = torch.maximum(nd1, nd2)
            non_terminal = (~dones_t).float()
            target_energy = energy_t + self.gamma * non_terminal * next_energy
            target_deadline = deadline_t + self.gamma * non_terminal * next_deadline

        e1, d1 = self.critic_1(states_t, actions_t, rhos_t)
        e2, d2 = self.critic_2(states_t, actions_t, rhos_t)
        loss_1 = F.smooth_l1_loss(e1, target_energy) + F.smooth_l1_loss(
            d1, target_deadline
        )
        loss_2 = F.smooth_l1_loss(e2, target_energy) + F.smooth_l1_loss(
            d2, target_deadline
        )
        _assert_finite_tensor("coordination_critic_loss", loss_1 + loss_2)
        grad_norms = []
        for optimizer, network, loss in (
            (self.optimizer_1, self.critic_1, loss_1),
            (self.optimizer_2, self.critic_2, loss_2),
        ):
            optimizer.zero_grad()
            loss.backward()
            grad = nn.utils.clip_grad_norm_(network.parameters(), self.max_grad_norm)
            optimizer.step()
            grad_norms.append(float(grad.item() if torch.is_tensor(grad) else grad))
        self.update_targets()
        self.learn_updates += 1
        return {
            "critic_1_loss": float(loss_1.item()),
            "critic_2_loss": float(loss_2.item()),
            "energy_q_mean": float(torch.maximum(e1, e2).detach().mean().item()),
            "deadline_q_mean": float(
                torch.maximum(d1, d2).detach().mean().item()
            ),
            "energy_target_mean": float(target_energy.mean().item()),
            "deadline_target_mean": float(target_deadline.mean().item()),
            "energy_td_abs_mean": float(
                (target_energy - e1.detach()).abs().mean().item()
            ),
            "deadline_td_abs_mean": float(
                (target_deadline - d1.detach()).abs().mean().item()
            ),
            "grad_norm_before_clip": float(np.mean(grad_norms)),
        }

    def update_targets(self, tau=None):
        tau = self.tau if tau is None else float(tau)
        for target, online in (
            (self.target_1, self.critic_1),
            (self.target_2, self.critic_2),
        ):
            for target_param, online_param in zip(
                target.parameters(), online.parameters()
            ):
                target_param.data.copy_(
                    tau * online_param.data + (1.0 - tau) * target_param.data
                )

    def state_dict(self, include_replay=False):
        payload = {
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_1": self.target_1.state_dict(),
            "target_2": self.target_2.state_dict(),
            "optimizer_1": self.optimizer_1.state_dict(),
            "optimizer_2": self.optimizer_2.state_dict(),
            "config": copy.deepcopy(self.config),
            "observed_transitions": int(self.observed_transitions),
            "learn_updates": int(self.learn_updates),
        }
        if include_replay:
            payload["replay"] = self.replay.state_dict()
        return payload

    def load_state_dict(self, payload):
        payload = dict(payload or {})
        self.critic_1.load_state_dict(payload["critic_1"])
        self.critic_2.load_state_dict(payload["critic_2"])
        self.target_1.load_state_dict(payload.get("target_1", payload["critic_1"]))
        self.target_2.load_state_dict(payload.get("target_2", payload["critic_2"]))
        if "optimizer_1" in payload:
            self.optimizer_1.load_state_dict(payload["optimizer_1"])
        if "optimizer_2" in payload:
            self.optimizer_2.load_state_dict(payload["optimizer_2"])
        self.observed_transitions = int(payload.get("observed_transitions", 0))
        self.learn_updates = int(payload.get("learn_updates", 0))
        if "replay" in payload:
            self.replay.load_state_dict(payload["replay"])
