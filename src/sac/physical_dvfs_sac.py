"""Low-level DVFS SAC with a fixed physical reward scale (no replay z-score).

Motivation (measured on
``results/hierarchical_reliability_aware_freeze_high_ep300_train_sac/20260826_142653``):

The production low reward is ``w_E * z(energy_saving_J) - w_T * z(slack_time_cost)``
where both z-scores use the *rolling replay standard deviation*.  Expressed as
an exchange rate -- how many joules one second of extra execution time is
allowed to cost -- that reward drifted as follows::

    episode    rho_mean   sigma_E(J)   sigma_T     implied price
        1        0.9488       25.07    0.00696      48.9 J/s
      101        0.7573      144.63    0.29187       6.9 J/s
      301        0.9480       87.95    0.01190      97.7 J/s
      600        0.9600       60.13    0.00707      88.2 J/s

The true marginal rate implied by ``E(rho) = (L / f_max) * (P_s / rho + P_d rho^2)``
is ``dE/dt = P_s - 2 * P_d * rho^3``, i.e. -714 J/s on ``cloud_fast`` and
-75 J/s on ``edge_efficient`` at ``rho = 1``.  Because the replay standard
deviation of ``slack_time_cost`` collapses as the policy converges to
``rho = 1``, the price of delay diverges, which pushes the policy further
towards ``rho = 1``: a self-reinforcing collapse.  The observed policy
(``rho = 0.873`` on ``cloud_fast``, ``0.956`` on ``edge_efficient``) is exactly
optimal for the *distorted* exchange rate.

This agent removes the replay statistics from the reward entirely.  Both terms
are dimensionless with a fixed scale::

    r = w_E * (E(1) - E(rho)) / E(1)          # in [0, 1), the ratio the upper
                                              # reward already uses
        - w_T * dt / max(slack_before, f * D) # already dimensionless
        - lambda_D * deadline_constraint_cost # unchanged counterfactual cost
        - lambda_R * reliability_violation    # unchanged

Nothing else about SAC changes: same networks, same replay, same optimizers,
same Lagrangian controllers, same state and action spaces.  With ``w_E = w_T``
the stationary point of the first two terms is::

    rho^3 = (P_s + w_T / w_E * tau * (P_s + P_d)) / (2 * P_d),  tau = T_exec(1) / slack

which reproduces the physical optimum ``rho* = (P_s / 2 P_d)^(1/3)`` as the
slack grows, and correctly returns ``rho = 1`` for ``edge_lowend`` where DVFS
costs energy.
"""
from __future__ import annotations

from src.sac.hierarchical_low_sac import LowLevelDVFSContinuousSACAgent

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class PhysicalDVFSLowSACAgent(LowLevelDVFSContinuousSACAgent):
    """SAC whose energy/time trade-off has a fixed, physical exchange rate."""

    def _base_reward(self, components, statistics):
        # ``statistics`` is deliberately unused: any replay-derived scale would
        # re-introduce the drift documented above.
        del statistics
        weights = torch.as_tensor(
            self.reward_weights, dtype=torch.float32, device=self.device
        )
        return (
            weights[0] * components[:, 0]
            - weights[1] * components[:, 1]
            + weights[2] * components[:, 2]
        )

    def remember_transition(
        self,
        state,
        action_rho,
        reward_components,
        next_state,
        done,
        *,
        critic_state=None,
        next_critic_state=None,
    ):
        """Store the dimensionless energy ratio in the energy component slot.

        ``build_low_sac_reward_components`` already returns
        ``energy_saving_ratio = (E(1) - E(rho)) / E(1)`` next to the raw joule
        value.  The ``sub_deadline_lagrangian_v1`` profile uses the same
        substitution, so the replay layout is unchanged.
        """
        components = dict(reward_components)
        for key in ("energy_saving_ratio", "reliability_budget_share_spent"):
            if key not in components:
                raise KeyError(
                    f"PhysicalDVFSLowSACAgent requires {key!r} in the low reward "
                    "components."
                )
        ratio = float(components["energy_saving_ratio"])
        components["energy_saving"] = ratio
        components["energy_saving_normalized"] = ratio
        # The production reliability cost is a *projected* violation that stays
        # at zero while the projection still assumes the remaining tasks are
        # 0.999 reliable, so DVFS can spend the whole workflow risk budget
        # before the signal reacts (measured: rho ~ 0.83 with a 58% reliability
        # success rate while the per-step cost stayed near zero).  Charge the
        # share of the workflow risk budget this action actually burned:
        #     (log R_task(1) - log R_task(rho)) / (-log R_required)
        # It is positive for every rho < 1 and sums over a workflow to the
        # fraction of the budget DVFS consumed, so the Lagrangian constrains
        # exactly the quantity that decides feasibility.
        components["reliability_violation"] = float(
            components["reliability_budget_share_spent"]
        )
        super().remember_transition(
            state,
            action_rho,
            components,
            next_state,
            done,
            critic_state=critic_state,
            next_critic_state=next_critic_state,
        )
