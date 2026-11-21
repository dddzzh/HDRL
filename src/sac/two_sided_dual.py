"""Two-sided multipliers for the shared Deadline price and the lower reliability price.

Deadline: lambda_D <- clip(lambda_D + step * (window miss rate - budget), 0, cap).
Reliability: lambda_R <- clip(lambda_R + step_up/down * clip(worst ratio - target, -1, 1), 0, cap),
where worst ratio is the batch maximum of -log(R_w) / -log(R_req) (1.0 = exactly at R_req).
"""
from collections import deque
import math


MODE = "two-sided-window-miss-rate-and-worst-hazard-ratio"
SETTINGS = dict(
    deadline_budget=0.05, deadline_window_episodes=10, deadline_step=1.0, deadline_cap=20.0,
    reliability_target_ratio=0.9, reliability_step_up=0.5, reliability_step_down=0.125,
    reliability_signal_clip=1.0, reliability_cap=20.0,
)


def worst_hazard_ratio(workflows):
    """Largest share of the reliability hazard budget used by any workflow in the batch."""
    ratios = []
    for w in workflows:
        rel, req = float(w["reliability"]), float(w["reliability_req"])
        if not (0. <= rel <= 1. and 0. < req < 1.):
            raise ValueError("Workflow reliability and requirement must lie in [0, 1] and (0, 1).")
        ratios.append(-math.log(max(rel, 1e-300)) / -math.log(req))
    if not ratios:
        raise ValueError("Worst hazard ratio requires a non-empty batch.")
    return max(ratios)


class TwoSidedDuals:
    def __init__(self, settings=SETTINGS, deadline=0., reliability=0., window=()):
        self.s = dict(settings)
        self.deadline = float(deadline)
        self.reliability = float(reliability)
        self.window = deque(((int(m), int(n)) for m, n in window), maxlen=self.s["deadline_window_episodes"])
        if not (0. <= self.deadline <= self.s["deadline_cap"] and 0. <= self.reliability <= self.s["reliability_cap"]):
            raise ValueError("Restored multipliers lie outside [0, cap].")

    def update(self, missed, workflow_count, worst_ratio):
        if workflow_count <= 0 or not 0 <= missed <= workflow_count or not math.isfinite(worst_ratio):
            raise ValueError("Invalid dual observation.")
        s = self.s
        self.window.append((int(missed), int(workflow_count)))
        rate = sum(m for m, _ in self.window) / sum(n for _, n in self.window)
        d_signal = rate - s["deadline_budget"]
        self.deadline = min(s["deadline_cap"], max(0., self.deadline + s["deadline_step"] * d_signal))
        clip = s["reliability_signal_clip"]
        r_signal = min(clip, max(-clip, worst_ratio - s["reliability_target_ratio"]))
        step = s["reliability_step_up"] if r_signal > 0 else s["reliability_step_down"]
        self.reliability = min(s["reliability_cap"], max(0., self.reliability + step * r_signal))
        return dict(deadline_window_miss_rate=rate, deadline_dual_signal=d_signal,
                    worst_hazard_ratio=worst_ratio, reliability_dual_signal=r_signal)

    def state(self):
        return dict(deadline=self.deadline, reliability=self.reliability, window=list(self.window))


ADAPTIVE_MODE = "parity-calibrated-conditional-integration-validation-miss-rate-and-windowed-worst-hazard-ratio"
ADAPTIVE_SETTINGS = dict(
    deadline_initialization="parity at the calibration validation: median Q_E action spread / median Q_D action spread",
    deadline_budget=0.05, deadline_log_step=0.2, deadline_signal_min=-1.0, deadline_signal_max=1.0,
    deadline_progress_history=3,
    reliability_target_ratio=0.9, reliability_cost_budget_fraction=0.9, reliability_window_episodes=10,
    reliability_log_step_up=0.5, reliability_log_step_down=0.25, reliability_signal_clip=1.0,
    initial_price=1.0, price_floor=1e-3, price_ceiling=1e6,
)


class AdaptiveDuals:
    """Scale-free prices with conditional integration (anti-windup), no hand-picked cap.

    lambda_D: calibrated once from the learned heads (energy and Deadline equally influential). Then, after
    each deterministic validation with miss rate m: lower it if m < budget; hold it while m is below the mean
    of the previous `deadline_progress_history` validations (the policy is still improving); otherwise raise
    it by at most ~22%.
    lambda_R: every `reliability_window_episodes` episodes, with w = mean batch-worst hazard ratio: lower it if
    w < target; hold it if w improved on the previous window; otherwise raise it.
    The lower reliability cost uses the same budget fraction as the target, so the price and the cost act at
    the same point. Floor/ceiling are numerical guards only.
    """

    def __init__(self, settings=ADAPTIVE_SETTINGS, deadline=None, reliability=None, calibrated=False,
                 miss_history=(), reliability_buffer=(), reliability_previous_mean=None):
        self.s = dict(settings)
        self.calibrated = bool(calibrated)
        self.deadline = float(self.s["initial_price"] if deadline is None else deadline)
        self.reliability = float(self.s["initial_price"] if reliability is None else reliability)
        self.miss_history = [float(x) for x in miss_history]
        self.reliability_buffer = [float(x) for x in reliability_buffer]
        self.reliability_previous_mean = (None if reliability_previous_mean is None
                                          else float(reliability_previous_mean))
        for value in (self.deadline, self.reliability):
            if not self.s["price_floor"] <= value <= self.s["price_ceiling"]:
                raise ValueError("Restored adaptive price lies outside the numerical guards.")

    def _scale(self, price, step, signal):
        return min(self.s["price_ceiling"], max(self.s["price_floor"], price * math.exp(step * signal)))

    @staticmethod
    def _rate(missed, workflow_count):
        if workflow_count <= 0 or not 0 <= missed <= workflow_count:
            raise ValueError("Invalid validation miss observation.")
        return missed / workflow_count

    def calibrate_deadline(self, energy_spread, deadline_spread, missed, workflow_count):
        if not (math.isfinite(energy_spread) and math.isfinite(deadline_spread)
                and energy_spread > 0 and deadline_spread > 0):
            raise ValueError("Parity calibration needs positive finite action spreads.")
        rate = self._rate(missed, workflow_count)
        before = self.deadline
        self.deadline = min(self.s["price_ceiling"], max(self.s["price_floor"], energy_spread / deadline_spread))
        self.calibrated = True
        self.miss_history = [rate]
        return dict(parity_calibration=True, action="calibrate", energy_action_spread=energy_spread,
                    deadline_action_spread=deadline_spread, validation_miss_rate=rate,
                    lambda_before=before, lambda_after=self.deadline)

    def update_deadline(self, missed, workflow_count):
        s = self.s
        rate = self._rate(missed, workflow_count)
        before, reference = self.deadline, None
        history = self.miss_history[-s["deadline_progress_history"]:]
        if rate < s["deadline_budget"]:
            action = "lower"
            signal = max(s["deadline_signal_min"], (rate - s["deadline_budget"]) / s["deadline_budget"])
            self.deadline = self._scale(self.deadline, s["deadline_log_step"], signal)
        elif len(history) < s["deadline_progress_history"]:
            action, signal = "hold_collecting_history", 0.
        elif rate < (reference := sum(history) / len(history)):
            action, signal = "hold_improving", 0.
        else:
            action = "raise"
            signal = min(s["deadline_signal_max"], (rate - s["deadline_budget"]) / s["deadline_budget"])
            self.deadline = self._scale(self.deadline, s["deadline_log_step"], signal)
        self.miss_history = (self.miss_history + [rate])[-s["deadline_progress_history"]:]
        return dict(action=action, validation_miss_rate=rate, progress_reference=reference,
                    deadline_dual_signal=signal, lambda_before=before, lambda_after=self.deadline)

    def update_reliability(self, worst_ratio, *, alternating_scale=None):
        if not math.isfinite(worst_ratio):
            raise ValueError("Invalid worst hazard ratio.")
        s = self.s
        if alternating_scale is not None and (alternating_scale not in ('small', 'middle')
                or s["reliability_window_episodes"] < 2):
            raise ValueError("Worst-scale reliability requires alternating Small/Medium episodes.")
        self.reliability_buffer.append(float(worst_ratio))
        info = dict(worst_hazard_ratio=worst_ratio, action="accumulate")
        if len(self.reliability_buffer) < s["reliability_window_episodes"]:
            return info
        mean = sum(self.reliability_buffer) / len(self.reliability_buffer)
        if alternating_scale is not None:
            # The last observation is the current scale; the checkpoint stores this window in order.
            current = self.reliability_buffer[::-2]
            other = self.reliability_buffer[-2::-2]
            other_scale = 'middle' if alternating_scale == 'small' else 'small'
            by_scale = {alternating_scale: sum(current) / len(current), other_scale: sum(other) / len(other)}
            info.update(pooled_window_mean=mean, window_mean_by_scale=by_scale)
            mean = max(by_scale.values())
        previous, before = self.reliability_previous_mean, self.reliability
        clip = s["reliability_signal_clip"]
        signal = min(clip, max(-clip, mean - s["reliability_target_ratio"]))
        if signal < 0:
            action = "lower"
            self.reliability = self._scale(self.reliability, s["reliability_log_step_down"], signal)
        elif previous is not None and mean < previous:
            action, signal = "hold_improving", 0.
        else:
            action = "raise"
            self.reliability = self._scale(self.reliability, s["reliability_log_step_up"], signal)
        self.reliability_previous_mean, self.reliability_buffer = mean, []
        info.update(action=action, window_mean=mean, previous_window_mean=previous,
                    reliability_dual_signal=signal, lambda_before=before, lambda_after=self.reliability)
        return info

    def state(self):
        return dict(deadline=self.deadline, reliability=self.reliability, calibrated=self.calibrated,
                    miss_history=list(self.miss_history), reliability_buffer=list(self.reliability_buffer),
                    reliability_previous_mean=self.reliability_previous_mean)


if __name__ == "__main__":
    a = AdaptiveDuals()
    info = a.calibrate_deadline(0.3, 0.001, 800, 800)
    assert a.calibrated and math.isclose(a.deadline, 300.) and a.miss_history == [1.]
    assert a.update_deadline(700, 800)["action"] == "hold_collecting_history"
    assert a.update_deadline(600, 800)["action"] == "hold_collecting_history"
    assert a.update_deadline(500, 800)["action"] == "hold_improving" and math.isclose(a.deadline, 300.)
    info = a.update_deadline(700, 800)                # stalled: above mean(0.875, 0.75, 0.625)
    assert info["action"] == "raise" and math.isclose(a.deadline, 300. * math.exp(0.2))
    info = a.update_deadline(0, 800)                  # all on time
    assert info["action"] == "lower" and math.isclose(a.deadline, 300.)
    for w in [1.2] * 9:
        assert a.update_reliability(w)["action"] == "accumulate"
    info = a.update_reliability(1.2)                  # first window above target, no reference: raise
    assert info["action"] == "raise" and math.isclose(a.reliability, math.exp(0.5 * 0.3))
    for w in [1.0] * 10:
        info = a.update_reliability(w)
    assert info["action"] == "hold_improving" and math.isclose(a.reliability, math.exp(0.15))
    for w in [0.8] * 10:
        info = a.update_reliability(w)
    assert info["action"] == "lower" and math.isclose(a.reliability, math.exp(0.15 - 0.25 * 0.1))
    a.update_reliability(0.95)
    assert AdaptiveDuals(**a.state()).state() == a.state()
    d = TwoSidedDuals()
    d.update(12, 20, 5.0)                         # early: many misses, reliability violated
    assert math.isclose(d.deadline, 0.55) and math.isclose(d.reliability, 0.5)
    for _ in range(100):
        d.update(15, 20, 5.0)
    assert d.deadline == 20. and d.reliability == 20.  # both capped
    for _ in range(10):
        d.update(0, 20, 0.6)                       # window flushed: all on time, worst at 60% of budget
    before = (d.deadline, d.reliability)
    info = d.update(0, 20, 0.6)
    assert math.isclose(info["deadline_window_miss_rate"], 0.) and math.isclose(d.deadline, before[0] - 0.05)
    assert math.isclose(d.reliability, before[1] - 0.125 * 0.3)
    info = d.update(1, 20, 0.9)                    # at budget and target: rate 0.005 < 0.05, ratio signal 0
    assert info["deadline_dual_signal"] < 0 and info["reliability_dual_signal"] == 0.
    r = TwoSidedDuals(deadline=d.deadline, reliability=d.reliability, window=d.state()["window"])
    assert r.state() == d.state()
    rows = [dict(reliability=0.96, reliability_req=0.95), dict(reliability=0.95, reliability_req=0.95)]
    assert math.isclose(worst_hazard_ratio(rows), 1.0)
    print("Two-sided dual checks passed.")
