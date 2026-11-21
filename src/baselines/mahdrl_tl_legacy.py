"""MAHDRL-TL-style legacy adapter using the author's masked Double-DQN learner.

The public repository contains the three-agent model, not a runnable TL ablation.
This adapter retains TA and replaces host/VM allocation with direct node selection.
Physical scheduling, energy, deadlines and reliability come from our existing env.
"""
from itertools import product
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/MAHDRL"
sys.path.insert(0, str(UPSTREAM))
from mahdrl.agents import D3QNAgent
from mahdrl.config import TA_HIDDEN, TRUNK
from mahdrl.learning import validate_transition
from mahdrl.networks import _polyak_update
from mahdrl.replay import behavior_probabilities
from mahdrl.workflow import weight_candidates

from src.sac.layerwise_excess import added_excess
from src.sac.two_sided_dual import ADAPTIVE_SETTINGS, worst_hazard_ratio
from src.simulator.edge_cloud_env import EdgeCloudWorkflowEnv
from src.simulator.metrics import compute_summary
from src.simulator.reliability import frequency_dependent_failure_rate
from src.simulator.resource import compute_execution_time

ACTION_TABLE = np.asarray(list(product((-.1, 0., .1), repeat=5)), dtype=np.float32)
UPSTREAM_COMMIT = "f3fe02139d4b9351eeec7bdda3cb9248fc048027"
FORMAT = "mahdrl_tl_legacy_reward_penalty_v2"


def placement_risk_cost(hazard, remaining_budget, workflow_budget, workflow_count):
    """Marginal absolute risk excess; the SAC DVFS-only difference is zero at rho=1."""
    if not (math.isfinite(hazard) and hazard >= 0 and math.isfinite(remaining_budget)
            and workflow_budget > 0 and workflow_count > 0):
        raise ValueError("Invalid placement risk budget.")
    return added_excess(hazard, 0., remaining_budget) / (workflow_budget * workflow_count)


class ReliabilityD3QN(D3QNAgent):
    """Author's single Q learner with a reliability penalty in the reward.

    Replay retains (base reward, risk cost) so sampled rewards use the current price.
    """
    def __init__(self, input_dim, output_dim, *, role, batch_size=256, device="cpu"):
        super().__init__(input_dim, output_dim, lr=3e-4, gamma=.95, batch_size=batch_size,
            buffer_size=60000 if role == "TA" else 120000,
            eps_decay_steps=40000 if role == "TA" else 80000,
            target_update_tau=.01 if role == "TA" else .005,
            hidden_dims=TA_HIDDEN if role == "TA" else TRUNK, device=torch.device(device))
        self.role = role
        self.risk_price = 0.

    def select_action(self, state, mask, deterministic=False, count_step=True, performance_times=None):
        legal = np.flatnonzero(np.asarray(mask) > .5)
        if not len(legal):
            raise ValueError("Cannot choose an action with no legal candidate.")
        if count_step:
            self._action_calls += 1
        if self.role == "TA" and not deterministic and random.random() < self.epsilon():
            return int(np.random.choice(legal))
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=self.device)[None]
            q = self.online(x)[0]
            if self.role == "VA" and not deterministic:
                if performance_times is None:
                    raise ValueError("Upstream guided allocation exploration needs candidate durations.")
                probabilities = behavior_probabilities(q.cpu().numpy(), mask, performance_times, self.epsilon())
                return int(np.random.choice(self.output_dim, p=probabilities))
            q = q.masked_fill(torch.as_tensor(mask <= .5, device=self.device), -torch.inf)
            return int(q.argmax())

    def remember_components(self, state, mask, action, base, risk, following, next_mask, done):
        validate_transition(state, mask, action, base, following, next_mask, done)
        if not math.isfinite(risk) or risk < 0:
            raise ValueError("Risk cost must be finite and nonnegative.")
        self.buffer.append((np.array(state, copy=True), np.array(mask, copy=True), int(action),
            (float(base), float(risk)), np.array(following, copy=True),
            np.array(next_mask, copy=True), bool(done)))
        self.f11.transition(done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, m, a, components, ns, nm, done = zip(*random.sample(self.buffer, self.batch_size))
        def tensor(x):
            return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        s, ns, nm, signals = map(tensor, (s, ns, nm, components))
        actions = torch.as_tensor(a, dtype=torch.int64, device=self.device)[:, None]
        live = 1. - tensor(done)
        reward = signals[:, 0] - self.risk_price * signals[:, 1]
        with torch.no_grad():
            best = self.online(ns).masked_fill(nm <= .5, -torch.inf).argmax(1, keepdim=True)
            active = live * (nm > .5).any(1)
            future = active * self.gamma * self.target(ns).gather(1, best)[:, 0]
            target = reward + future
        selected_q = self.online(s).gather(1, actions)[:, 0]
        loss = F.smooth_l1_loss(selected_q, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite MAHDRL-TL loss.")
        self.f11.target(future, target)
        self.f11.numerics.observe("Q_selected", selected_q, scalar=True)
        self.f11.numerics.observe("TD_target", target, scalar=True)
        self.f11.numerics.observe("TD_error", target - selected_q, scalar=True)
        self.f11.loss(loss)
        self.optim.zero_grad()
        loss.backward()
        self.f11.numerics.gradients(self.online.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip, error_if_nonfinite=True)
        self.optim.step()
        _polyak_update(self.target, self.online, self.target_update_tau)
        self._updates += 1
        self.f11.updates += 1
        self._eps_steps += 1
        return dict(loss=float(loss.detach()), mean_reward=float(reward.mean()))

    def snapshot(self):
        return dict(online=self.online.state_dict(), target=self.target.state_dict(),
            optim=self.optim.state_dict(),
            replay=list(self.buffer), updates=self._updates, eps_steps=self._eps_steps,
            action_calls=self._action_calls)

    def restore(self, saved):
        for key in ("online", "target", "optim"):
            getattr(self, key).load_state_dict(saved[key])
        self.buffer.clear()
        self.buffer.extend(saved["replay"])
        self._updates, self._eps_steps = saved["updates"], saved["eps_steps"]
        self._action_calls = saved["action_calls"]


class LegacyTLScene:
    def __init__(self, workflows, resources, simulation, seed):
        self.env = EdgeCloudWorkflowEnv(workflows, resources, simulation, seed)
        self.weights = np.full(5, .2, dtype=np.float32)
        self.max_ready = 1
        self.n = len(workflows)
        self.budget = -math.log(float(simulation["workflow_reliability_requirement"]))
        exponent = float(simulation.get("reliability_exponent", 3.))
        self.reserve_per_cycle = max(frequency_dependent_failure_rate(s, 1., exponent)
            * compute_execution_time(1., s, 1.) for s in self.env.servers)

    def idle_mask(self):
        return np.asarray([self.env._server_execution_available_time(s) <= self.env.current_time + 1e-9
                           for s in self.env.servers], dtype=np.float32)

    def ensure_decision(self):
        # Wait for an actual event, retaining the paper's idle-only dispatch rule.
        while not self.env.done:
            if self.env._get_ready_tasks() and self.idle_mask().any():
                return True
            if self.env.event_queue.empty():
                self.env._update_done()
                if not self.env.done:
                    raise RuntimeError("Unfinished workflow with no events or legal dispatch.")
                break
            self.env._process_next_event()
            self.env._process_events_through(self.env.current_time)
            self.env._update_done()
        return False

    def task_attributes(self, ready):
        frequency = np.mean([s["max_freq_hz"] for s in self.env.servers])
        return np.asarray([[t["task_ready_time"], t["workload_cycles"] / frequency,
            t["upward_rank"], t["absolute_deadline"] - self.env.current_time - t["workload_cycles"] / frequency,
            t["absolute_deadline"]] for t in ready], dtype=np.float64)

    def manager_state(self):
        ready = self.env._get_ready_tasks()
        self.max_ready = max(self.max_ready, len(ready))
        released = [w for w in self.env.workflow_states if w["released"]]
        statuses = [s for w in released for s in w["task_status"].values()]
        pending = sum(s not in ("finished", "running") for s in statuses)
        waits = [t["ready_queue_waiting_time"] for t in ready] or [0.]
        time_ref = max([self.env.workflow_relative_deadline(w) for w in released] or [1.])
        busy = 1. - self.idle_mask()
        capacities = np.asarray([s["max_freq_hz"] for s in self.env.servers])
        slacks = self.task_attributes(ready)[:, 3] if ready else np.array([0.])
        context = [len(ready) / self.max_ready, max(0, pending-len(ready)) / max(1, len(statuses)),
            busy.mean(), np.mean(waits)/time_ref, max(waits)/time_ref, 1.-busy.mean(),
            float(busy @ capacities / capacities.sum()), busy.mean(),
            slacks.min()/time_ref, slacks.mean()/time_ref]
        return np.r_[np.asarray(context, np.float32), self.weights].astype(np.float32)

    def manager_mask(self):
        return weight_candidates(self.weights, ACTION_TABLE)[1].astype(np.float32)

    def ordered_tasks(self, action):
        choices, valid = weight_candidates(self.weights, ACTION_TABLE)
        if not valid[action]:
            raise ValueError("Illegal task-order weight increment.")
        self.weights = choices[action].copy()
        ready = self.env._get_ready_tasks()
        raw = self.task_attributes(ready)
        span = raw.max(0) - raw.min(0)
        ascending = (raw - raw.min(0))
        descending = raw.max(0) - raw
        numerator = np.where(np.array([-1., -1., 1., -1., -1.]) > 0, ascending, descending)
        normalized = np.divide(numerator, span, out=np.zeros_like(raw), where=span > 0)
        scores = normalized @ self.weights
        return [ready[i] for i in sorted(range(len(ready)), key=lambda i:
            (-scores[i], ready[i]["task_ready_time"], str(ready[i]["workflow_id"]), ready[i]["task_id"]))]

    def budget_values(self, task):
        state = self.env.workflow_by_id[str(task["workflow_id"])]
        risks = state["task_reliability"]
        used = -math.fsum(math.log(max(float(r), 1e-300)) for r in risks.values())
        future = self.reserve_per_cycle * sum(t.workload_cycles for t in state["workflow"].tasks
            if t.id != task["task_id"] and t.id not in risks)
        remaining = ADAPTIVE_SETTINGS["reliability_cost_budget_fraction"] * self.budget - used - future
        return used, future, remaining

    def node_state(self, task):
        context = self.env.build_hierarchical_decision_context(task["workflow_id"], task["task_id"])
        safety = context["server_safety_metrics"]
        used, future, remaining = self.budget_values(task)
        ddl = task["relative_deadline"]
        features = []
        idle = self.idle_mask()
        for i, server in enumerate(self.env.servers):
            m = safety[str(server["id"])]
            features.extend([server["max_freq_hz"]/1e9,
                (server["static_power"]+server["dynamic_power"])/1000.,
                server.get("bandwidth_mbps", 100.)/10000., m["execution_time_rho_1"]/ddl,
                max(0., m["estimated_ast"]-self.env.current_time)/ddl,
                (m["predicted_workflow_finish_rho_1"]-task["arrival_time"])/ddl,
                m["primary_energy_ref_rho1"]/1000., idle[i],
                -math.log(max(m["task_reliability_rho_1"], 1e-300))/self.budget])
        task_features = [task["workload_cycles"]/1e10, task["upward_rank"]/ddl,
            task["ready_queue_waiting_time"]/ddl, (task["absolute_deadline"]-self.env.current_time)/ddl,
            task["workflow_task_count"]/100., used/self.budget, future/self.budget, remaining/self.budget]
        obs = np.r_[self.manager_state(), np.tanh(task_features), np.tanh(features)].astype(np.float32)
        if not np.isfinite(obs).all():
            raise FloatingPointError("Non-finite legacy TL observation.")
        return obs, idle, safety, remaining


def build_agents(node_count, *, batch_size=256, device="cpu"):
    return dict(TA=ReliabilityD3QN(15, len(ACTION_TABLE), role="TA", batch_size=batch_size, device=device),
                VA=ReliabilityD3QN(23+9*node_count, node_count, role="VA", batch_size=batch_size, device=device))


def run_episode(scene, agents, *, training=False, risk_price=0., warmup=.03):
    env = scene.env
    for agent in agents.values():
        agent.risk_price = risk_price
    pending = None
    phases, risk_total, base_return = 0, 0., 0.
    before_updates = {role: a._updates for role, a in agents.items()}
    def learn(role, transition):
        agent = agents[role]
        if training:
            agent.remember_components(*transition)
            if len(agent.buffer) >= max(agent.batch_size, int(agent.buffer.maxlen*warmup)):
                agent.update()
    while scene.ensure_decision():
        s_ta, m_ta = scene.manager_state(), scene.manager_mask()
        a_ta = agents["TA"].select_action(s_ta, m_ta, deterministic=not training, count_step=training)
        phase_energy = phase_delay = phase_risk = 0.
        assigned = 0
        for task in scene.ordered_tasks(a_ta):
            if not scene.idle_mask().any():
                break
            s, mask, safety, remaining = scene.node_state(task)
            if pending is not None:
                learn("VA", (*pending, s, mask, False))
            times = np.asarray([max(0., safety[str(server["id"])]["estimated_ast"]-env.current_time)
                + safety[str(server["id"])]["execution_time_rho_1"] for server in env.servers])
            action = agents["VA"].select_action(s, mask, deterministic=not training,
                count_step=training, performance_times=times)
            server = env.servers[action]
            _, _, _, info = env.step(dict(workflow_id=task["workflow_id"], task_id=task["task_id"],
                server_id=server["id"], rho=1.))
            event = info["event"]
            if event["rho"] != 1. or event.get("backup_enabled"):
                raise AssertionError("TL must use full frequency and no backup.")
            hazard = event["failure_rate"] * event["exec_time"]
            cost = placement_risk_cost(hazard, remaining, scene.budget, scene.n)
            limit = safety[str(server["id"])]["task_latest_finish_time"]
            delay = max(0., event["logical_finish_time"]-limit)
            reward = -.25*.001*event["energy"] - .75*min(1., delay/300.)
            pending = (s, mask, action, reward, cost)
            phase_energy += event["energy"]
            phase_delay += delay
            phase_risk += cost
            assigned += 1
        if not assigned:
            raise RuntimeError("Empty dispatch phase despite legal tasks and nodes.")
        ongoing = scene.ensure_decision()
        phase_reward = -.25*.001*phase_energy - .75*phase_delay/300.
        next_ta = scene.manager_state() if ongoing else np.zeros_like(s_ta)
        next_mask = scene.manager_mask() if ongoing else np.zeros_like(m_ta)
        learn("TA", (s_ta, m_ta, a_ta, phase_reward, phase_risk, next_ta, next_mask, not ongoing))
        risk_total += phase_risk
        base_return += phase_reward
        phases += 1
    if pending is not None:
        learn("VA", (*pending, np.zeros_like(pending[0]), np.zeros_like(pending[1]), True))
    workflows = env._build_workflow_results()
    if not env.done or any(w["finish_time"] is None for w in workflows):
        raise AssertionError("TL episode did not complete every workflow.")
    metrics = compute_summary(workflows, env.event_log, env.servers)
    missed = sum(not w["deadline_satisfied"] for w in workflows)
    rel_missed = sum(not w["reliability_met"] for w in workflows)
    return dict(energy_J=metrics["total_energy"], rho=1., workflow_count=len(workflows),
        deadline_missed_workflows=missed, reliability_missed_workflows=rel_missed,
        ESP90=missed/len(workflows) <= .1+1e-9, CR=max(0., worst_hazard_ratio(workflows)-1.),
        worst_hazard_ratio=worst_hazard_ratio(workflows), risk_cost_total=risk_total,
        base_return=base_return, penalized_return=base_return-risk_price*risk_total,
        phases=phases, task_count=len(env.event_log),
        updates={role: a._updates-before_updates[role] for role, a in agents.items()},
        epsilon={role: a.epsilon() for role, a in agents.items()})
