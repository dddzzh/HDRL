from __future__ import annotations

import copy
from collections import deque

import numpy as np

from src.D3QN.D3QN import DuelingDeepQNetwork
from src.D3QN.buffer import ReplayBuffer
from src.D3QN.dual_head_constrained import DualHeadHierarchicalD3QNAgent
from src.D3QN.hierarchical_d3qn_agent import HighLevelD3QNAgent
from src.device_utils import resolve_agent_device

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


EPS = 1e-8
CALIBRATION_BINS = np.asarray(
    [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
     0.6, 0.7, 0.8, 0.9, 1.0 + 1e-8],
    dtype=np.float64,
)


class WorkflowRiskReplay:
    """Completed-workflow replay with equal expected weight per workflow.

    Each stored item contains all selected state/action pairs from one completed
    workflow and its terminal binary miss label. Sampling first chooses a
    workflow and then a step, so long workflows do not dominate the risk loss.
    """

    def __init__(
        self,
        state_dim,
        max_transitions=50000,
        batch_size=256,
        positive_fraction=0.3,
        min_workflows=32,
        seed=42,
    ):
        self.state_dim = int(state_dim)
        self.max_transitions = int(max_transitions)
        self.batch_size = int(batch_size)
        self.positive_fraction = float(positive_fraction)
        self.min_workflows = int(min_workflows)
        if self.max_transitions <= 0 or self.batch_size <= 0:
            raise ValueError("Risk replay sizes must be positive.")
        if not 0.0 <= self.positive_fraction <= 1.0:
            raise ValueError("positive_fraction must be in [0, 1].")
        if self.min_workflows <= 0:
            raise ValueError("min_workflows must be positive.")
        self.rng = np.random.default_rng(int(seed))
        self.workflows = deque()
        self.transition_count = 0
        self.completed_count = 0

    @property
    def workflow_count(self):
        return len(self.workflows)

    @property
    def ready(self):
        return self.workflow_count >= self.min_workflows

    def add_workflow(self, states, actions, label):
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        label = float(label)
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError("Workflow risk states have an invalid shape.")
        if actions.shape != (states.shape[0],) or states.shape[0] <= 0:
            raise ValueError("Workflow risk actions do not match states.")
        if not np.isfinite(states).all() or label not in {0.0, 1.0}:
            raise ValueError("Workflow risk replay contains invalid values.")
        item = {
            "states": states.copy(),
            "actions": actions.copy(),
            "label": label,
        }
        self.workflows.append(item)
        self.transition_count += int(states.shape[0])
        self.completed_count += 1
        while self.transition_count > self.max_transitions and len(self.workflows) > 1:
            removed = self.workflows.popleft()
            self.transition_count -= int(removed["states"].shape[0])

    def _class_positions(self):
        labels = np.asarray(
            [float(item["label"]) for item in self.workflows], dtype=np.float64
        )
        return np.flatnonzero(labels > 0.5), np.flatnonzero(labels <= 0.5)

    def sample(self):
        if not self.ready:
            return None
        positive, negative = self._class_positions()
        total_workflows = max(self.workflow_count, 1)
        if positive.size and negative.size:
            positive_count = int(round(self.batch_size * self.positive_fraction))
            positive_count = min(max(positive_count, 1), self.batch_size - 1)
            negative_count = self.batch_size - positive_count
            selected = np.concatenate(
                [
                    self.rng.choice(positive, positive_count, replace=True),
                    self.rng.choice(negative, negative_count, replace=True),
                ]
            ).astype(np.int64)
            # Correct the class-quota sampling back to uniform-workflow risk.
            natural_positive = positive.size / total_workflows
            natural_negative = negative.size / total_workflows
            sampled_positive = positive_count / self.batch_size
            sampled_negative = negative_count / self.batch_size
            weights = np.concatenate(
                [
                    np.full(
                        positive_count,
                        natural_positive / max(sampled_positive, EPS),
                    ),
                    np.full(
                        negative_count,
                        natural_negative / max(sampled_negative, EPS),
                    ),
                ]
            ).astype(np.float32)
        else:
            selected = self.rng.choice(
                self.workflow_count, self.batch_size, replace=True
            ).astype(np.int64)
            weights = np.ones(self.batch_size, dtype=np.float32)

        order = self.rng.permutation(self.batch_size)
        selected = selected[order]
        weights = weights[order]
        weights /= max(float(weights.mean()), EPS)
        states = np.empty((self.batch_size, self.state_dim), dtype=np.float32)
        actions = np.empty(self.batch_size, dtype=np.int64)
        labels = np.empty(self.batch_size, dtype=np.float32)
        for row, workflow_position in enumerate(selected):
            item = self.workflows[int(workflow_position)]
            step = int(self.rng.integers(0, item["states"].shape[0]))
            states[row] = item["states"][step]
            actions[row] = item["actions"][step]
            labels[row] = float(item["label"])
        return {
            "states": states,
            "actions": actions,
            "labels": labels,
            "weights": weights,
        }

    def state_dict(self):
        return {
            "state_dim": self.state_dim,
            "max_transitions": self.max_transitions,
            "batch_size": self.batch_size,
            "positive_fraction": self.positive_fraction,
            "min_workflows": self.min_workflows,
            "transition_count": self.transition_count,
            "completed_count": self.completed_count,
            "workflows": list(self.workflows),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, payload):
        if int(payload["state_dim"]) != self.state_dim:
            raise ValueError("Risk replay state dimension does not match.")
        self.workflows = deque(copy.deepcopy(payload.get("workflows", [])))
        self.transition_count = int(payload.get("transition_count", 0))
        self.completed_count = int(payload.get("completed_count", len(self.workflows)))
        if payload.get("rng_state") is not None:
            self.rng.bit_generator.state = copy.deepcopy(payload["rng_state"])


def _weighted_calibration_metrics(probabilities, labels, weights):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weight_sum = max(float(weights.sum()), EPS)
    brier = float(np.sum(weights * np.square(probabilities - labels)) / weight_sum)
    ece = 0.0
    occupied = 0
    for lower, upper in zip(CALIBRATION_BINS[:-1], CALIBRATION_BINS[1:]):
        member = (probabilities >= lower) & (probabilities < upper)
        if not np.any(member):
            continue
        bucket_weight = float(weights[member].sum())
        predicted = float(np.sum(weights[member] * probabilities[member]) / bucket_weight)
        observed = float(np.sum(weights[member] * labels[member]) / bucket_weight)
        ece += bucket_weight / weight_sum * abs(predicted - observed)
        occupied += 1
    positives = labels > 0.5
    positive_weight = float(weights[positives].sum())
    recall = (
        float(weights[positives & (probabilities >= 0.5)].sum()) / positive_weight
        if positive_weight > EPS
        else 0.0
    )
    return brier, float(ece), float(recall), int(occupied)


class FeasibleRiskDualHeadD3QN:
    """Energy Double-DQN plus a workflow-terminal Deadline risk classifier."""

    def __init__(
        self,
        *,
        state_dim,
        action_dim,
        learning_rate=3e-4,
        risk_learning_rate=None,
        hidden_dim=256,
        hidden_layers=2,
        energy_gamma=0.99,
        tau=0.005,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=1.8e-4,
        replay_size=50000,
        batch_size=256,
        max_grad_norm=1.0,
        risk_positive_fraction=0.3,
        risk_min_workflows=32,
        risk_gate_min_completed_workflows=500,
        risk_threshold_start=0.2,
        risk_threshold_end=0.01,
        risk_anneal_start_episode=100,
        risk_anneal_end_episode=500,
        device="auto",
        seed=42,
    ):
        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch is required for feasible-risk D3QN.")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.energy_gamma = float(energy_gamma)
        self.tau = float(tau)
        self.epsilon = float(epsilon)
        self.eps_min = float(epsilon_min)
        self.eps_dec = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.risk_gate_min_completed_workflows = int(
            risk_gate_min_completed_workflows
        )
        self.risk_threshold_start = float(risk_threshold_start)
        self.risk_threshold_end = float(risk_threshold_end)
        self.risk_anneal_start_episode = int(risk_anneal_start_episode)
        self.risk_anneal_end_episode = int(risk_anneal_end_episode)
        if not 0.0 <= self.risk_threshold_end <= self.risk_threshold_start <= 1.0:
            raise ValueError("Risk thresholds must satisfy 0 <= end <= start <= 1.")
        if self.risk_anneal_end_episode <= self.risk_anneal_start_episode:
            raise ValueError("Risk threshold anneal interval is invalid.")
        if self.risk_gate_min_completed_workflows < 0:
            raise ValueError("risk_gate_min_completed_workflows must be non-negative.")
        self.episodes_seen = 0
        self.rng = np.random.default_rng(int(seed))
        self.device = resolve_agent_device(device)

        network_args = dict(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=int(hidden_dim),
            hidden_layers=int(hidden_layers),
        )
        self.energy_q_eval = DuelingDeepQNetwork(**network_args).to(self.device)
        self.energy_q_target = DuelingDeepQNetwork(**network_args).to(self.device)
        self.risk_logit_net = DuelingDeepQNetwork(**network_args).to(self.device)
        self.energy_optimizer = torch.optim.Adam(
            self.energy_q_eval.parameters(), lr=float(learning_rate)
        )
        risk_lr = float(learning_rate) if risk_learning_rate is None else float(
            risk_learning_rate
        )
        self.risk_optimizer = torch.optim.Adam(
            self.risk_logit_net.parameters(), lr=risk_lr
        )
        self.optimizer = self.energy_optimizer
        self.memory = ReplayBuffer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_size=int(replay_size),
            batch_size=self.batch_size,
        )
        self.risk_replay = WorkflowRiskReplay(
            state_dim=self.state_dim,
            max_transitions=int(replay_size),
            batch_size=self.batch_size,
            positive_fraction=float(risk_positive_fraction),
            min_workflows=int(risk_min_workflows),
            seed=int(seed) + 137,
        )
        self.update_network_parameters(tau=1.0)

    @property
    def current_risk_threshold(self):
        episode = int(self.episodes_seen)
        if episode <= self.risk_anneal_start_episode:
            return self.risk_threshold_start
        if episode >= self.risk_anneal_end_episode:
            return self.risk_threshold_end
        fraction = (
            (episode - self.risk_anneal_start_episode)
            / (self.risk_anneal_end_episode - self.risk_anneal_start_episode)
        )
        return float(
            self.risk_threshold_start
            + fraction * (self.risk_threshold_end - self.risk_threshold_start)
        )

    @property
    def risk_gate_active(self):
        return (
            self.risk_replay.completed_count
            >= self.risk_gate_min_completed_workflows
        )

    def update_network_parameters(self, tau=None):
        tau = self.tau if tau is None else float(tau)
        for target, online in zip(
            self.energy_q_target.parameters(), self.energy_q_eval.parameters()
        ):
            target.data.copy_(tau * online.data + (1.0 - tau) * target.data)

    def decrement_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)

    def action_values(self, observation):
        state = torch.as_tensor(
            np.asarray([observation], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            energy = self.energy_q_eval(state)[0]
            risk = torch.sigmoid(self.risk_logit_net(state)[0])
        return (
            energy.detach().cpu().numpy().astype(np.float64),
            risk.detach().cpu().numpy().astype(np.float64),
        )

    def _greedy_action_numpy(self, energy, risk, action_mask):
        energy = np.asarray(energy, dtype=np.float64)
        risk = np.asarray(risk, dtype=np.float64)
        action_mask = np.asarray(action_mask, dtype=bool)
        valid = np.flatnonzero(action_mask)
        if valid.size <= 0:
            raise ValueError("Feasible-risk D3QN action mask contains no valid action.")
        if energy.shape != (self.action_dim,) or risk.shape != (self.action_dim,):
            raise ValueError("Feasible-risk action values have invalid shapes.")
        if not np.isfinite(energy).all() or not np.isfinite(risk).all():
            raise ValueError("Feasible-risk action values contain NaN or Inf.")
        threshold = self.current_risk_threshold
        feasible = action_mask & (risk <= threshold) if self.risk_gate_active else action_mask
        no_feasible = not bool(feasible.any())
        if no_feasible:
            action = int(valid[np.argmin(risk[valid])])
        else:
            action = int(np.argmax(np.where(feasible, energy, -np.inf)))
        return action, feasible, no_feasible

    def choose_action(self, observation, action_mask, *, is_train=True):
        energy, risk = self.action_values(observation)
        greedy, feasible, no_feasible = self._greedy_action_numpy(
            energy, risk, action_mask
        )
        valid = np.flatnonzero(np.asarray(action_mask, dtype=bool))
        explored = bool(is_train) and self.rng.random() < self.epsilon
        action = int(self.rng.choice(valid)) if explored else greedy
        valid_risk = risk[valid]
        risk_rank = int(np.argsort(valid_risk).tolist().index(
            int(np.flatnonzero(valid == action)[0])
        )) + 1
        diagnostics = {
            "risk_threshold": float(self.current_risk_threshold),
            "risk_gate_active": float(self.risk_gate_active),
            "risk_mean": float(valid_risk.mean()),
            "risk_min": float(valid_risk.min()),
            "risk_selected": float(risk[action]),
            "risk_selected_rank": float(risk_rank),
            "feasible_action_count": float(feasible.sum()),
            "no_feasible_action": float(no_feasible),
            "risk_exploration_action": float(explored),
        }
        return action, energy, risk, diagnostics

    def remember_energy(
        self,
        state,
        action,
        energy_reward,
        next_state,
        done,
        action_mask,
        next_action_mask,
    ):
        self.memory.store_transition(
            state,
            action,
            energy_reward,
            next_state,
            done,
            action_mask,
            next_action_mask,
            base_energy_reward=energy_reward,
            deadline_constraint_cost=0.0,
        )

    def remember_workflow_risk(self, states, actions, miss):
        self.risk_replay.add_workflow(states, actions, miss)

    def _next_actions(self, next_states, next_masks):
        with torch.no_grad():
            energy = self.energy_q_eval(next_states)
            risk = torch.sigmoid(self.risk_logit_net(next_states))
            if self.risk_gate_active:
                feasible = next_masks & (risk <= self.current_risk_threshold)
            else:
                feasible = next_masks
            has_feasible = feasible.any(dim=-1)
            energy_action = torch.argmax(
                energy.masked_fill(~feasible, -1e9), dim=-1
            )
            risk_action = torch.argmin(
                risk.masked_fill(~next_masks, 1e9), dim=-1
            )
            return torch.where(has_feasible, energy_action, risk_action)

    def _learn_energy(self):
        if not self.memory.ready():
            return None
        indices = self.rng.choice(
            self.memory.size, self.batch_size, replace=False
        ).astype(np.int64)
        states = torch.as_tensor(
            self.memory.states[indices], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            self.memory.actions[indices], dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            self.memory.base_energy_rewards[indices],
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.as_tensor(
            self.memory.next_states[indices], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            self.memory.dones[indices], dtype=torch.bool, device=self.device
        )
        next_masks = torch.as_tensor(
            self.memory.next_action_masks[indices],
            dtype=torch.bool,
            device=self.device,
        )
        row = torch.arange(self.batch_size, device=self.device)
        q_value = self.energy_q_eval(states)[row, actions]
        with torch.no_grad():
            next_actions = self._next_actions(next_states, next_masks)
            next_q = self.energy_q_target(next_states)[row, next_actions]
            no_valid = next_masks.sum(dim=-1) <= 0
            next_q = torch.where(dones | no_valid, torch.zeros_like(next_q), next_q)
            target = rewards + self.energy_gamma * next_q
        loss = F.smooth_l1_loss(q_value, target)
        self.energy_optimizer.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            self.energy_q_eval.parameters(), self.max_grad_norm
        )
        self.energy_optimizer.step()
        td = (target - q_value).detach()
        return {
            "energy_loss": float(loss.item()),
            "energy_q_mean": float(q_value.detach().mean().item()),
            "energy_target_mean": float(target.detach().mean().item()),
            "energy_td_abs_mean": float(td.abs().mean().item()),
            "energy_grad_norm_before_clip": float(grad),
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std(unbiased=False).item()),
        }

    def _learn_risk(self):
        batch = self.risk_replay.sample()
        if batch is None:
            return None
        states = torch.as_tensor(
            batch["states"], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            batch["actions"], dtype=torch.long, device=self.device
        )
        labels = torch.as_tensor(
            batch["labels"], dtype=torch.float32, device=self.device
        )
        weights = torch.as_tensor(
            batch["weights"], dtype=torch.float32, device=self.device
        )
        row = torch.arange(self.batch_size, device=self.device)
        logits = self.risk_logit_net(states)[row, actions]
        losses = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        loss = (losses * weights).mean()
        self.risk_optimizer.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            self.risk_logit_net.parameters(), self.max_grad_norm
        )
        self.risk_optimizer.step()
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        brier, ece, recall, occupied = _weighted_calibration_metrics(
            probabilities, batch["labels"], batch["weights"]
        )
        return {
            "deadline_loss": float(loss.item()),
            "risk_probability_mean": float(np.mean(probabilities)),
            "risk_label_mean": float(
                np.average(batch["labels"], weights=batch["weights"])
            ),
            "risk_brier_score": brier,
            "risk_ece": ece,
            "risk_positive_recall": recall,
            "risk_calibration_occupied_bins": float(occupied),
            "risk_grad_norm_before_clip": float(grad),
            "risk_positive_replay_fraction": float(
                np.mean(batch["labels"] > 0.5)
            ),
        }

    def learn(self):
        energy = self._learn_energy()
        risk = self._learn_risk()
        if energy is None and risk is None:
            return {}
        if energy is not None:
            self.update_network_parameters()
        self.decrement_epsilon()
        metrics = {}
        metrics.update(energy or {})
        metrics.update(risk or {})
        metrics["loss"] = float(
            metrics.get("energy_loss", 0.0) + metrics.get("deadline_loss", 0.0)
        )
        metrics["risk_threshold"] = float(self.current_risk_threshold)
        metrics["risk_gate_active"] = float(self.risk_gate_active)
        return metrics

    def state_dict(self, *, include_replay=False):
        return {
            "energy_q_eval": self.energy_q_eval.state_dict(),
            "energy_q_target": self.energy_q_target.state_dict(),
            "risk_logit_net": self.risk_logit_net.state_dict(),
            "energy_optimizer": self.energy_optimizer.state_dict(),
            "risk_optimizer": self.risk_optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episodes_seen": self.episodes_seen,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "energy_replay": self.memory.state_dict() if include_replay else None,
            "risk_replay": self.risk_replay.state_dict() if include_replay else None,
            "risk_completed_count": self.risk_replay.completed_count,
        }

    def load_state_dict(self, payload):
        self.energy_q_eval.load_state_dict(payload["energy_q_eval"])
        self.energy_q_target.load_state_dict(payload["energy_q_target"])
        self.risk_logit_net.load_state_dict(payload["risk_logit_net"])
        self.energy_optimizer.load_state_dict(payload["energy_optimizer"])
        self.risk_optimizer.load_state_dict(payload["risk_optimizer"])
        self.epsilon = float(payload.get("epsilon", self.epsilon))
        self.episodes_seen = int(payload.get("episodes_seen", self.episodes_seen))
        if payload.get("rng_state") is not None:
            self.rng.bit_generator.state = copy.deepcopy(payload["rng_state"])
        if payload.get("energy_replay") is not None:
            self.memory.load_state_dict(payload["energy_replay"])
        if payload.get("risk_replay") is not None:
            self.risk_replay.load_state_dict(payload["risk_replay"])
        else:
            self.risk_replay.completed_count = int(
                payload.get("risk_completed_count", 0)
            )


class FeasibleRiskHighLevelD3QNAgent(HighLevelD3QNAgent):
    def __init__(self, config=None, seed=42):
        super().__init__(config=config, seed=seed)
        config = dict(config or {})
        self.constraint_handling_mode = "feasible_workflow_risk"
        self.agent_R = FeasibleRiskDualHeadD3QN(
            state_dim=self.encoder.high_state_dim,
            action_dim=self.action_dim,
            learning_rate=float(config.get("learning_rate", 3e-4)),
            risk_learning_rate=float(
                config.get("risk_learning_rate", config.get("learning_rate", 3e-4))
            ),
            hidden_dim=int(config.get("hidden_dim", 256)),
            hidden_layers=int(config.get("hidden_layers", 2)),
            energy_gamma=float(config.get("gamma", config.get("reward_gamma", 0.99))),
            tau=float(config.get("tau", 0.005)),
            epsilon=float(config.get("epsilon", 1.0)),
            epsilon_min=float(config.get("epsilon_min", 0.1)),
            epsilon_decay=float(config.get("epsilon_decay", 1.8e-4)),
            replay_size=int(config.get("replay_size", 50000)),
            batch_size=int(config.get("batch_size", 256)),
            max_grad_norm=float(config.get("max_grad_norm", 1.0)),
            risk_positive_fraction=float(config.get("risk_positive_fraction", 0.3)),
            risk_min_workflows=int(config.get("risk_min_workflows", 32)),
            risk_gate_min_completed_workflows=int(
                config.get("risk_gate_min_completed_workflows", 500)
            ),
            risk_threshold_start=float(config.get("risk_threshold_start", 0.2)),
            risk_threshold_end=float(config.get("risk_threshold_end", 0.01)),
            risk_anneal_start_episode=int(
                config.get("risk_anneal_start_episode", 100)
            ),
            risk_anneal_end_episode=int(
                config.get("risk_anneal_end_episode", 500)
            ),
            device=config.get("device", "auto"),
            seed=seed,
        )

    def select_assignment(
        self,
        env_state,
        deterministic=False,
        selector_mode="policy",
        action_values=None,
        selection_diagnostics=None,
    ):
        if selector_mode != "policy":
            raise ValueError("Feasible-risk D3QN supports only policy selection.")
        if action_values is not None:
            raise ValueError("External action re-ranking is disabled.")
        observation = self.build_high_state(env_state)
        context = self._require_context(env_state)
        primary_mask, backup_pair_mask = self._context_masks(context)
        action, energy_values, risk_values, risk_diagnostics = (
            self.agent_R.choose_action(
                observation.vector,
                primary_mask,
                is_train=not bool(deterministic),
            )
        )
        primary_idx = self.decode_action(action)
        decision = self._decision_payload(
            observation,
            context,
            primary_mask,
            backup_pair_mask,
            primary_idx,
            None,
            action,
            float(energy_values[action]),
            selector_mode,
            np.zeros(self.max_servers, dtype=bool),
            False,
        )
        decision.update(
            {
                "energy_q_value": float(energy_values[action]),
                "deadline_risk": float(risk_values[action]),
                "deadline_q_value": float(risk_values[action]),
                "combined_q_value": float(energy_values[action]),
                "action_deadline_lambda": 0.0,
                **risk_diagnostics,
                **dict(selection_diagnostics or {}),
            }
        )
        return decision

    @staticmethod
    def workflow_terminal_labels(buffer):
        labels = {}
        for workflow_id, indices in buffer.workflow_groups():
            final_index = indices[-1]
            deadline = max(float(buffer.deadline_limits[final_index]), EPS)
            completion = float(buffer.partial_makespans[final_index])
            labels[str(workflow_id)] = float(completion > deadline + EPS)
        return labels

    def remember_from_high_buffer(self, buffer):
        if buffer.size == 0:
            return {
                "stored": 0.0,
                "binary_deadline_cost_total": 0.0,
                "binary_missed_workflows": 0.0,
            }
        labels = self.workflow_terminal_labels(buffer)
        zero_state = np.zeros(self.encoder.high_state_dim, dtype=np.float32)
        zero_mask = np.zeros(self.action_dim, dtype=bool)
        for index in range(buffer.size):
            done = index == buffer.size - 1
            next_state = zero_state if done else buffer.states[index + 1]
            next_mask = zero_mask if done else np.asarray(
                buffer.primary_masks[index + 1], dtype=bool
            ).copy()
            action = self.encode_action(buffer.primary_actions[index])
            energy_reward = float(buffer.infos[index]["high_base_energy_reward"])
            workflow_miss = float(labels[str(buffer.workflow_ids[index])])
            buffer.infos[index]["dual_head_energy_reward"] = energy_reward
            buffer.infos[index]["dual_head_workflow_miss"] = workflow_miss
            buffer.infos[index]["dual_head_deadline_risk_label"] = workflow_miss
            self.agent_R.remember_energy(
                buffer.states[index],
                action,
                energy_reward,
                next_state,
                done,
                np.asarray(buffer.primary_masks[index], dtype=bool).copy(),
                next_mask,
            )
        for workflow_id, indices in buffer.workflow_groups():
            self.agent_R.remember_workflow_risk(
                [buffer.states[index] for index in indices],
                [self.encode_action(buffer.primary_actions[index]) for index in indices],
                labels[str(workflow_id)],
            )
        return {
            "stored": float(buffer.size),
            "binary_deadline_cost_total": float(sum(labels.values())),
            "binary_missed_workflows": float(sum(labels.values())),
            "risk_stored_workflows": float(len(labels)),
        }

    def update(self, buffer=None):
        remember = self.remember_from_high_buffer(buffer) if buffer is not None else {
            "stored": 0.0,
            "binary_deadline_cost_total": 0.0,
            "binary_missed_workflows": 0.0,
            "risk_stored_workflows": 0.0,
        }
        self.agent_R.episodes_seen += 1
        totals = {
            "replay_size": float(self.replay.size),
            "risk_replay_workflows": float(self.agent_R.risk_replay.workflow_count),
            "risk_replay_transitions": float(self.agent_R.risk_replay.transition_count),
            "risk_completed_workflows": float(self.agent_R.risk_replay.completed_count),
            "epsilon": float(self.epsilon),
            "stored_transitions": float(remember["stored"]),
            "risk_stored_workflows": float(remember.get("risk_stored_workflows", 0.0)),
            "deadline_reward_penalty_total": 0.0,
            "deadline_cost_reward_weight": 0.0,
            "binary_deadline_cost_total": float(remember["binary_deadline_cost_total"]),
            "binary_missed_workflows": float(remember["binary_missed_workflows"]),
        }
        updates = 0
        for _ in range(self.training_per_episode):
            metrics = self.agent_R.learn()
            if not metrics:
                continue
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            updates += 1
        fixed = {
            "replay_size",
            "risk_replay_workflows",
            "risk_replay_transitions",
            "risk_completed_workflows",
            "epsilon",
            "stored_transitions",
            "risk_stored_workflows",
            "deadline_reward_penalty_total",
            "deadline_cost_reward_weight",
            "binary_deadline_cost_total",
            "binary_missed_workflows",
        }
        if updates:
            for key in list(totals):
                if key not in fixed:
                    totals[key] /= updates
        totals["updates"] = float(updates)
        totals["epsilon"] = float(self.epsilon)
        totals["risk_threshold"] = float(self.agent_R.current_risk_threshold)
        totals["risk_gate_active"] = float(self.agent_R.risk_gate_active)
        return totals


class FeasibleRiskHierarchicalD3QNAgent(DualHeadHierarchicalD3QNAgent):
    """Standalone fixed-rho=1 wrapper preserving the v1 implementation."""

    CHECKPOINT_FORMAT = "hierarchical_dual_head_feasible_risk_d3qn_v2"

    def __init__(self, config=None, seed=42):
        super().__init__(config=config, seed=seed)
        shared = dict(self.config.get("shared", {}))
        high_config = {**shared, **dict(self.config.get("high", {}))}
        self.high_agent = FeasibleRiskHighLevelD3QNAgent(high_config, seed=seed)

    def load_high_checkpoint(self, path, *, freeze=False):
        payload = self.load_checkpoint(path)
        if freeze:
            networks = (
                self.high_agent.agent_R.energy_q_eval,
                self.high_agent.agent_R.energy_q_target,
                self.high_agent.agent_R.risk_logit_net,
            )
            for network in networks:
                network.eval()
                for parameter in network.parameters():
                    parameter.requires_grad_(False)
        return payload
