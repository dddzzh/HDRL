# All-experiment independent-EMA paper refresh

Read original CSVs and cross-check JSONL. Only the 800 DQN-Edge records change: main 540, Alibaba 60, workflow count 100, arrival rate 100. Other algorithms retain 2400 unchanged records. Ablations and framework figure are unchanged. Raw files are not overwritten.

Main workload seeds: 0--19. Supplementary workload seeds: 742000--742019. Each DQN-Edge realization starts from the saved checkpoint EMA and updates it within that realization.

Arrival 0.20/s: DQN-Edge ESP@90=95%, Rel.=100%; seed 742011 has on-time fraction 0.86. All failures are retained. All other new DQN-Edge cells attain 100% for both metrics.

Main energy inference: 81 exact two-sided paired Wilcoxon tests, with family-wide Bonferroni and BH corrections. Wilson intervals describe batch-level attainment and are not Wilcoxon tests. Inference remains conditional on the reported workloads and fixed checkpoints; it is not a population or training-repeat guarantee.

Backup: `paper\manuscript\backups\independent_ema_all_20261031`.

Plot reproduction:

- `python plot/plot_current_generalization_energy.py --independent-ema --separators`
- `python plot/plot_alibaba_energy_only.py --independent-ema --separators`
- `python plot/plot_workload_sensitivity_options.py --independent-ema --single-column`
