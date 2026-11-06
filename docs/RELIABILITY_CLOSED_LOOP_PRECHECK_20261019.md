# Reliability closed-loop precheck — frozen before execution

User authorized this no-training precheck on 2026-10-19. No short training is authorized by this launch.

## Research Question Card

- Question: Can an original-action-floor-compliant frequency intervention reduce energy while meeting ESP@90 but violating workflow reliability in a fully recomputed closed-loop schedule?
- Type: exploratory validation mechanism precheck, not a causal learning ablation.
- Hypothesis: loose deadlines may expose a frequency/reliability conflict hidden by the full policy's high executed frequencies.
- Existing evidence: results/reliability_diagnostic_20261015/FINDINGS.md and rho_ee_reference/README.md show fixed-placement reliability risk, but do not compute new deadline outcomes.
- Missing evidence: deadline and reliability jointly measured after new scheduling, queues and execution durations.
- Support: a predeclared rule has lower energy than rho=1 on the same validation seed, deadline_success_rate >= .9 and reliability_success_rate < 1.
- Weaken: no such conflict for any predeclared rule. This does not prove no other policy can exhibit a conflict.
- Minimal action: exactly 25 episodes below, retain all results and stop. Never choose extra seeds/rules to induce failure.

## Protocol

- Resource: 74 servers, 56 edge + 18 cloud; resource seed 42.
- Full SM74 model: original 20261009_115939/best_d3qn_checkpoint.pt, SHA256 32764207f74371ae00c0c3cfdf9c16bd760c0a3143be560dd7f72f2ec1c652ea. No model reselection.
- Original eval config: configs/v3_sm_74_new_eval_50wf.json; only in-memory scale=large and deadline=loose, 50 workflows, .03/s.
- Validation seeds: 542000–542004, the first five from checkpoint.config.training.eval_seed=542000 / eval_episodes=20. These are known validation samples, not an untouched final holdout. The test seed range 742000–742019 is not used.
- Rules: max(1.0, action floor), max(.8, floor), max(.7, floor), max(.6, floor), and per-server action floor (rho_ee with physical/agent lower bounds).
- The rho_safe quantity is not a hard clip in the existing SAC selector; do not add a new deadline shield. Keep production simulator and policies unchanged on disk.
- Each cell reloads the same checkpoint with initialization seed 742000. Original upper policy selects tasks/servers; frequency is overridden only for this diagnostic. Freezing weights does NOT fix placements: different durations change queues and future decisions.
- training=False, no_grad, low actor update mask false; hash neural weights before/after each cell. Do not update replay, optimizers or dual multipliers.
- All five rules for a seed have identical workflow/arrival/physical/deadline fingerprints, frozen before evaluation. No backup; same reliability requirement .9, same fault rates and compute model.
- 5 processes, Torch CPU threads 1 each. No duplicate run or automatic partial resume. All 25 cells required for completion.
- Save frozen source SHA, full metrics, per-workflow results and per-task actual frequency/reliability/energy in an independent results directory. Original main2.tex and main scenario matrix are hash-guarded, not edited.

## Output interpretation

ESP: fraction of five batches with at least 45/50 workflows on time. Rel: fraction of five batches with all 50 workflow reliabilities >= .9. Workflow attainment is separately labeled and cannot replace the batch metric.

Only a feasibility/conflict screen: a fixed-rho rule is not trained Reliability-off. Even a positive conflict does not guarantee a later ablation changes Rel. All failure cases and parameter settings remain visible.

Command: `python tools/precheck_reliability_closed_loop.py run`. Existing output directory refuses restart. `report` reads existing complete outputs only and starts no evaluation.
