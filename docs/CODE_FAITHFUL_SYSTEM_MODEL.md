# Code-Faithful System Model and Optimization Formulation

> **Purpose.** This document reconstructs the mathematical model implemented by the current formal training pipeline. It is the technical source of truth for drafting Section III of the paper. All equations below correspond to the formal configurations
> `v2_small_28edge_9cloud_dvfs_v3_future_esp.json` and
> `v2_medium_40edge_13cloud_dvfs_v3_future_esp.json`.
>
> **Scope boundary.** The formal method uses primary-only execution, continuous DVFS, computing-energy minimization, workflow-level Deadline constraints, and workflow-level reliability constraints. Backup duplication, task sub-Deadlines, transmission energy, device upload, and result download are disabled in the formal runs.

## 1. Main notation

| Symbol | Meaning |
|---|---|
| $\mathcal{W}$ | Set of workflows in an episode |
| $G_w=(\mathcal{V}_w,\mathcal{E}_w)$ | DAG of workflow $w$ |
| $i,j$ | Task indices |
| $k,h$ | Server indices |
| $L_{wi}$ | Workload of task $i$ in CPU cycles |
| $d_{wij}$ | Data size on dependency edge $(i,j)$ in MB |
| $A_w$ | Arrival time of workflow $w$ |
| $D_w^{\mathrm{rel}}$ | Relative workflow Deadline |
| $D_w^{\mathrm{abs}}=A_w+D_w^{\mathrm{rel}}$ | Absolute workflow Deadline |
| $M_w^{\mathrm{HEFT}}$ | Reference HEFT makespan at $\rho=1$ |
| $x_{wik}\in\{0,1\}$ | 1 if task $i$ of workflow $w$ is placed on server $k$ |
| $\rho_{wi}\in[0,1]$ | Normalized DVFS ratio selected for task $i$ |
| $f_k^{\max}$ | Maximum frequency of server $k$ |
| $P_k^{\mathrm{ind}},P_k^{\mathrm{dyn}}$ | Frequency-independent and maximum dynamic active powers |
| $B_{kh}$ | Bandwidth between servers $k$ and $h$ |
| $S_{wi},F_{wi}$ | Actual start and finish times of task $i$ |
| $C_w=F_w-A_w$ | Workflow response time |
| $R_{wi},R_w$ | Task and workflow reliability |
| $\lambda_D^H,\lambda_R^H$ | Upper Deadline and reliability multipliers |
| $\lambda_D^L,\lambda_R^L$ | Lower Deadline and reliability multipliers |

The superscripts $H$ and $L$ refer to the upper placement policy and lower DVFS policy, respectively.

## 2. Dynamic workflow model

Each workflow is a directed acyclic graph

$$
G_w=(\mathcal{V}_w,\mathcal{E}_w),
$$

where task $i\in\mathcal{V}_w$ requires $L_{wi}$ CPU cycles and edge $(i,j)\in\mathcal{E}_w$ transfers $d_{wij}$ MB from task $i$ to task $j$.

The workflow stream follows a Poisson process. The inter-arrival time is

$$
\Delta A_w=A_w-A_{w-1}\sim\operatorname{Exp}(\lambda_a),
\qquad \lambda_a=0.1\ \mathrm{s}^{-1},
$$

with $A_1=0$. Each episode contains 20 workflows.

The formal workload generator samples Pegasus DAG templates. For every generated task and dependency edge,

$$
L_{wi}\sim\mathcal{U}(0.5,20.0)\times 10^9\ \text{cycles},
$$

$$
d_{wij}\sim\mathcal{U}(1,30)\ \text{MB}.
$$

The environment exposes only ready tasks. A task is ready when the workflow has arrived and every parent task has finished. Among all ready tasks, the formal configuration fixes the dispatch order to the smallest workload first. Therefore, the learned upper policy chooses a server for the current task; it does not choose which ready task to dispatch.

## 3. Heterogeneous edge-cloud resource model

The server set is

$$
\mathcal{K}=\mathcal{K}^{\mathrm{edge}}\cup\mathcal{K}^{\mathrm{cloud}}.
$$

The Small scenario contains 28 edge and 9 cloud servers, while the Medium scenario contains 40 edge and 13 cloud servers. Edge servers are divided as evenly as possible among the efficient, legacy, and low-end profiles; cloud servers are divided between fast and standard profiles. Five heterogeneous profiles are sampled with $\pm5\%$ perturbations for frequency and active-power components and $\pm10\%$ perturbations for server bandwidth:

| Profile | Layer | $f^{\max}$ (GHz) | $P^{\mathrm{ind}}$ (W) | $P^{\mathrm{dyn}}$ (W) | $\lambda^{\max}$ (s$^{-1}$) |
|---|---:|---:|---:|---:|---:|
| edge-lowend | edge | 1.80 | 3.00 | 12.80 | $1.0\times10^{-5}$ |
| edge-efficient | edge | 2.40 | 5.87 | 36.03 | $1.0\times10^{-5}$ |
| edge-legacy | edge | 2.90 | 18.06 | 102.34 | $1.0\times10^{-5}$ |
| cloud-standard | cloud | 3.25 | 29.16 | 213.84 | $6.0\times10^{-6}$ |
| cloud-fast | cloud | 3.60 | 41.32 | 371.88 | $6.0\times10^{-6}$ |

The inter-server bandwidth is layer dependent:

$$
B_{kh}=
\begin{cases}
1000\ \mathrm{Mbps}, & k,h\in\mathcal{K}^{\mathrm{edge}},\\
10000\ \mathrm{Mbps}, & k,h\in\mathcal{K}^{\mathrm{cloud}},\\
100\ \mathrm{Mbps}, & \text{otherwise}.
\end{cases}
$$

Entry-task upload and exit-task download are outside the scheduling boundary. Only data carried by DAG edges contributes communication delay.

## 4. Communication, queueing, and task timing

If a parent task $i$ and child task $j$ run on different servers $k$ and $h$, their communication time is

$$
T_{wij}^{\mathrm{com}}(k,h)=
\begin{cases}
0, & k=h,\\
\dfrac{8d_{wij}}{B_{kh}}, & k\ne h.
\end{cases}
$$

Let $k(w,i)$ denote the selected server of task $i$. The data-ready time of task $i$ on candidate server $k$ is

$$
T_{wi,k}^{\mathrm{data}}
=
\begin{cases}
A_w, & \operatorname{pred}(i)=\varnothing,\\
\displaystyle
\max_{j\in\operatorname{pred}(i)}
\left(F_{wj}+T_{wji}^{\mathrm{com}}(k(w,j),k)\right),
& \text{otherwise}.
\end{cases}
$$

Let $Q_k(t)$ be the tail of server $k$'s non-preemptive execution queue at decision time $t$. The actual start time is

$$
S_{wi}=\max\left\{t,A_w,T_{wi,k}^{\mathrm{data}},Q_k(t)\right\}.
$$

The task finish time is

$$
F_{wi}=S_{wi}+T_{wi,k}^{\mathrm{exe}}(\rho_{wi}).
$$

After committing the task,

$$
Q_k\leftarrow F_{wi}.
$$

The realized queue delay charged to placement is

$$
q_{wi}=\left[S_{wi}-T_{wi,k}^{\mathrm{data}}\right]_+.
$$

The workflow completes when its last exit task finishes:

$$
F_w=\max_{i\in\mathcal{V}_w}F_{wi},
\qquad
C_w=F_w-A_w.
$$

## 5. DVFS execution-time and energy model

For server $k$, the normalized frequency ratio $\rho$ gives

$$
f_k(\rho)=\rho f_k^{\max}.
$$

The execution time of task $i$ is

$$
T_{wi,k}^{\mathrm{exe}}(\rho)
=\frac{L_{wi}}{\rho f_k^{\max}}.
$$

The active power model is cubic in normalized frequency:

$$
P_k(\rho)=P_k^{\mathrm{ind}}+P_k^{\mathrm{dyn}}\rho^3.
$$

Consequently, the computing energy is

$$
E_{wi,k}(\rho)
=P_k(\rho)T_{wi,k}^{\mathrm{exe}}(\rho)
=\frac{L_{wi}}{f_k^{\max}}
\left(
\frac{P_k^{\mathrm{ind}}}{\rho}
+P_k^{\mathrm{dyn}}\rho^2
\right).
$$

The always-on leakage that cannot be changed by scheduling is excluded from the optimization objective. Communication energy is also excluded. Thus, episode energy is

$$
E^{\mathrm{tot}}
=\sum_{w\in\mathcal{W}}
\sum_{i\in\mathcal{V}_w}
\sum_{k\in\mathcal{K}}
x_{wik}E_{wi,k}(\rho_{wi}).
$$

Differentiating the task-energy function gives the energy-efficient DVFS floor

$$
\rho_k^{\mathrm{ee}}
=
\left(
\frac{P_k^{\mathrm{ind}}}{2P_k^{\mathrm{dyn}}}
\right)^{1/3}.
$$

The executed action is restricted to

$$
\rho_{wi}\in
\left[
\underline{\rho}_k,1
\right],
\qquad
\underline{\rho}_k
=\max\left\{
\rho^{\mathrm{actor}}_{\min},
\rho_k^{\mathrm{hw}},
\rho_k^{\mathrm{ee}}
\right\},
$$

where $\rho^{\mathrm{actor}}_{\min}=0.2$ and $\rho_k^{\mathrm{hw}}=0.1$ in the formal configurations. Since all configured $\rho_k^{\mathrm{ee}}>0.2$, the energy-efficient floor is normally active.

## 6. Workflow Deadline construction

For each sampled workflow, HEFT is run on the same server pool and network at $\rho=1$. Its reference makespan is $M_w^{\mathrm{HEFT}}$. The relative Deadline is

$$
D_w^{\mathrm{rel}}=\mu_w M_w^{\mathrm{HEFT}},
\qquad
\mu_w\in\{2,3\},
$$

and the absolute Deadline is

$$
D_w^{\mathrm{abs}}=A_w+D_w^{\mathrm{rel}}.
$$

The probability of drawing the tighter multiplier $\mu_w=2$ is

$$
\Pr(\mu_w=2)=
\begin{cases}
0.2, & \text{Loose},\\
0.5, & \text{Medium},\\
0.8, & \text{Tight}.
\end{cases}
$$

A workflow satisfies its Deadline if

$$
C_w\le D_w^{\mathrm{rel}}
\quad\Longleftrightarrow\quad
F_w\le D_w^{\mathrm{abs}}.
$$

For episode $e$ with $N_e$ completed workflows, the Deadline success rate is

$$
\operatorname{DSR}_e
=\frac{1}{N_e}
\sum_{w=1}^{N_e}
\mathbb{I}\left(C_w\le D_w^{\mathrm{rel}}\right).
$$

Across $S$ evaluation seeds, the empirical success probability at threshold $g$ is

$$
\operatorname{ESP}@g
=\frac{1}{S}
\sum_{s=1}^{S}
\mathbb{I}\left(\operatorname{DSR}_s\ge g\right).
$$

The formal paper metric uses $g=0.90$.

## 7. Frequency-dependent reliability model

The maximum-frequency failure rate of server $k$ is $\lambda_k^{\max}$. The actual transient failure rate under DVFS is

$$
\lambda_k(\rho)
=\lambda_k^{\max}
10^{
d_k
\frac{f_k^{\max}-f_k(\rho)}
{f_k^{\max}-f_k^{\min}}
},
$$

where $d_k=3$ and $f_k^{\min}=\rho_k^{\mathrm{hw}}f_k^{\max}$. Equivalently,

$$
\lambda_k(\rho)
=\lambda_k^{\max}
10^{d_k(1-\rho)/(1-\rho_k^{\mathrm{hw}})}.
$$

Assuming an exponential failure process, the reliability of a primary task replica is

$$
R_{wi,k}(\rho)
=\exp\left(
-\lambda_k(\rho)T_{wi,k}^{\mathrm{exe}}(\rho)
\right).
$$

Because formal runs disable backup duplication, task reliability equals primary-replica reliability. The workflow reliability is

$$
R_w=\prod_{i\in\mathcal{V}_w}R_{wi,k(w,i)}(\rho_{wi}).
$$

The formal requirement is

$$
R_w\ge R_w^{\mathrm{req}}=0.9.
$$

The reliability success rate of an evaluation run is

$$
\operatorname{RSR}
=\frac{1}{N}
\sum_{w=1}^{N}
\mathbb{I}\left(R_w\ge R_w^{\mathrm{req}}\right).
$$

## 8. Constrained optimization problem

The ideal online scheduling problem is

$$
\begin{aligned}
\min_{\{x_{wik},\rho_{wi}\}}
\quad &
\mathbb{E}\left[E^{\mathrm{tot}}\right]\\
\text{s.t.}\quad
& \sum_{k\in\mathcal{K}}x_{wik}=1,
&& \forall w,i,\\
& x_{wik}\in\{0,1\},
&& \forall w,i,k,\\
& \underline{\rho}_k\le\rho_{wi}\le1
&& \text{if }x_{wik}=1,\\
& S_{wi}\ge F_{wj}+T_{wji}^{\mathrm{com}},
&& \forall(j,i)\in\mathcal{E}_w,\\
& \text{tasks assigned to one server do not overlap},\\
& C_w\le D_w^{\mathrm{rel}},
&& \forall w,\\
& R_w\ge R_w^{\mathrm{req}},
&& \forall w.
\end{aligned}
$$

The stochastic operational form used for learning and model selection replaces strict per-sample feasibility with distributional targets:

$$
\Pr\!\left(\operatorname{DSR}_e\ge0.90\right)\ge0.90,
$$

$$
\Pr\!\left(R_w\ge0.9\right)\approx1.
$$

The evaluation protocol reports both constraints separately and also reports their joint satisfaction rate.

## 9. Hierarchical decision process

At decision epoch $t$, the environment fixes the current ready task $(w_t,i_t)$. The hierarchical policy factorizes the joint scheduling action as

$$
\pi(k_t,\rho_t\mid s_t)
=\pi_H(k_t\mid s_t^H)
\pi_L(\rho_t\mid s_t^L,k_t).
$$

The upper policy selects one real server,

$$
k_t\sim\pi_H(\cdot\mid s_t^H),
$$

and the lower policy selects a continuous frequency ratio conditional on that server,

$$
\rho_t\sim\pi_L(\cdot\mid s_t^L,k_t).
$$

All real servers remain selectable. Deadline feasibility estimates are included in the state and reward; they are not used as a hard server mask.

## 10. Upper-level state

The upper state concatenates four current-task features, twelve features for each server, one validity mask per server, and eight global features:

$$
s_t^H=
\left[
\phi_t^{\mathrm{task}},
\phi_{t,1}^{\mathrm{srv}},\ldots,
\phi_{t,K}^{\mathrm{srv}},
m_1,\ldots,m_K,
\phi_t^{\mathrm{global}}
\right].
$$

Its dimension is

$$
\dim(s_t^H)=4+12K+K+8.
$$

Therefore, the formal Small and Medium states have 493 and 701 dimensions, respectively.

### 10.1 Current-task features

The four task features are normalized workload, remaining time from task readiness, remaining workflow time, and ready-queue waiting time:

$$
\phi_t^{\mathrm{task}}
=\left[
\tanh\left(\frac{L_{wi}}{10^{10}}\right),
\operatorname{clip}\left(\frac{D_w^{\mathrm{abs}}-T_{wi}^{\mathrm{ready}}}{D_w^{\mathrm{rel}}}\right),
\operatorname{clip}\left(\frac{D_w^{\mathrm{abs}}-t}{D_w^{\mathrm{rel}}}\right),
\operatorname{clip}\left(\frac{[t-T_{wi}^{\mathrm{ready}}]_+}{D_w^{\mathrm{rel}}}\right)
\right].
$$

### 10.2 Task-server features

For every candidate server, the twelve features encode

1. queued processing time;
2. estimated execution time at $\rho=1$;
3. estimated start delay;
4. estimated workflow-relative finish time;
5. safe execution budget;
6. a reserved communication-time slot;
7. safe-frequency estimate $\rho^{\mathrm{safe}}$;
8. task energy at $\rho=1$, normalized by the cross-server median;
9. task failure probability at $\rho=1$;
10. exponentially smoothed queued processing time;
11. recent primary-assignment intensity;
12. recent backup-assignment intensity, which remains zero when backup is disabled.

Time quantities are normalized by $D_w^{\mathrm{rel}}$ and passed through bounded transformations. In the current formal state, communication delay affects the candidate AST and predicted finish time. The separate reserved communication-time slot remains zero because the decision context does not copy a standalone transfer-time value into that field.

### 10.3 Global features

The eight global features are the ready-task ratio, server ratio, mean and minimum remaining-Deadline ratios, busy-server ratio, mean normalized server wait, global critical pressure, and worst-workflow critical pressure. For each active workflow $w$, let

$$
p_w=\frac{H_w^{\mathrm{remain}}}{[D_w^{\mathrm{abs}}-t]_+}
$$

be the ratio of its largest remaining HEFT rank to remaining wall-clock time. The two pressure features are

$$
p^{\mathrm{global}}=\tanh\left(\frac{\sum_w p_w}{K}\right),
\qquad
p^{\mathrm{worst}}=\frac{\operatorname{clip}(\max_w p_w,0,2)}{2}.
$$

## 11. Remaining-critical-path safety state

Let $r_u(w,i)$ be the HEFT upward rank of task $i$. Let

$$
\bar T_{wi}^{\mathrm{exe},1}
=\frac{1}{K}\sum_{k\in\mathcal{K}}
\frac{L_{wi}}{f_k^{\max}}.
$$

The estimated critical-path tail after the current task is

$$
H_{wi}^{\mathrm{tail}}
=\left[r_u(w,i)-\bar T_{wi}^{\mathrm{exe},1}\right]_+.
$$

For candidate server $k$, the latest safe finish of the current task is

$$
F_{wi,k}^{\mathrm{latest}}
=D_w^{\mathrm{abs}}-H_{wi}^{\mathrm{tail}}.
$$

With safety margin $\delta_s=0.05$, the safe execution budget is

$$
B_{wi,k}^{\mathrm{safe}}
=(1-\delta_s)
\left(F_{wi,k}^{\mathrm{latest}}-\widehat S_{wi,k}\right).
$$

The corresponding frequency estimate is

$$
\rho_{wi,k}^{D}
=\frac{T_{wi,k}^{\mathrm{exe}}(1)}
{B_{wi,k}^{\mathrm{safe}}},
$$

and

$$
\rho_{wi,k}^{\mathrm{safe}}
=\operatorname{clip}
\left(
\max\{\underline\rho_k,\rho_{wi,k}^{D}\},
\underline\rho_k,1
\right).
$$

If the safe budget is non-positive or $\rho_{wi,k}^{D}>1$, the diagnostic safe frequency is set to 1. This quantity is a state feature. The formal implementation does not hard-clip the SAC action to $\rho^{\mathrm{safe}}$; it only clips to the server-specific energy-efficient action floor.

## 12. Upper three-head Dueling Double DQN

The upper learner has three independent Dueling networks:

$$
Q_E(s,k),\qquad Q_D(s,k),\qquad Q_R(s,k),
$$

for energy reward, Deadline cost, and reliability cost. Each head uses

$$
Q_j(s,k)
=V_j(s)+A_j(s,k)
-\frac{1}{|\mathcal{K}|}
\sum_{h\in\mathcal{K}}A_j(s,h),
\qquad j\in\{E,D,R\}.
$$

The Lagrangian-combined action value is

$$
Q_{\mathrm{comb}}(s,k)
=Q_E(s,k)
-\lambda_D^H Q_D(s,k)
-\lambda_R^H Q_R(s,k).
$$

The greedy action is

$$
k_t^*=\arg\max_{k\in\mathcal{K}}Q_{\mathrm{comb}}(s_t,k),
$$

with $\epsilon$-greedy exploration during training.

For Double-DQN learning, the shared next action is selected by the online composite value,

$$
k_{t+1}^*
=\arg\max_k
\left[
Q_E^{\mathrm{on}}(s_{t+1},k)
-\lambda_D^H Q_D^{\mathrm{on}}(s_{t+1},k)
-\lambda_R^H Q_R^{\mathrm{on}}(s_{t+1},k)
\right].
$$

Each head keeps its own Bellman target:

$$
y_E=r_t^E+\gamma_E
Q_E^{\mathrm{tar}}(s_{t+1},k_{t+1}^*),
$$

$$
y_D=c_t^D+\gamma_D
Q_D^{\mathrm{tar}}(s_{t+1},k_{t+1}^*),
$$

$$
y_R=c_t^R+\gamma_R
Q_R^{\mathrm{tar}}(s_{t+1},k_{t+1}^*).
$$

The three Huber losses are optimized independently and their target networks use Polyak updates.

## 13. Upper energy reward

For the current task, let

$$
\bar E_{wi}^{1}
=\operatorname{median}_{k\in\mathcal{K}}
E_{wi,k}(1)
$$

be the cross-server median energy reference. The formal `ep740_lagrangian_v1` profile exposes realized energy after the lower SAC action to the upper learner. Thus,

$$
r_{t}^{E,H}
=\frac{w_E^H}{|\mathcal{V}_w|}
\frac{\bar E_{wi}^{1}-E_{wi,k_t}(\rho_t)}
{\bar E_{wi}^{1}},
\qquad w_E^H=1.
$$

This cooperative reward allows placement learning to account for the energy actually produced by the joint server-frequency decision.

## 14. Upper Deadline credit assignment

Let $C_{w,t}^{\mathrm{partial}}$ denote the largest realized finish time of workflow $w$ relative to its arrival after decision $t$. Its incremental normalized completion cost is

$$
i_t
=\frac{
\left[C_{w,t}^{\mathrm{partial}}
-C_{w,t-1}^{\mathrm{partial}}\right]_+
}{D_w^{\mathrm{rel}}}.
$$

The normalized queue credit is

$$
u_t=\frac{q_{wi}}{|\mathcal{V}_w|D_w^{\mathrm{rel}}}.
$$

The dense Deadline mass is

$$
d_t=i_t+\beta_q u_t.
$$

For a completed workflow, define

$$
m_w=\mathbb{I}(C_w>D_w^{\mathrm{rel}}).
$$

The workflow miss credit is distributed over its transitions according to their dense masses:

$$
c_{t,\mathrm{wf}}^D
=d_t
+m_w
\frac{d_t}{\sum_{\tau\in\mathcal{T}_w}d_\tau},
$$

with uniform allocation if the denominator is zero.

To align the Deadline head with checkpoint ESP, define the episode failure indicator

$$
e_D=\mathbb{I}(\operatorname{DSR}_e<0.90).
$$

For each decision, the predicted future-finish regret is

$$
g_t
=\frac{
\left[
\widehat F_{w,k_t}^{\mathrm{future}}
-\min_k\widehat F_{w,k}^{\mathrm{future}}
\right]_+
}{D_w^{\mathrm{rel}}}.
$$

The ESP-aligned episode credit is

$$
c_{t,\mathrm{ep}}^D
=e_D\frac{g_t}{\sum_\tau g_\tau},
$$

again using uniform allocation if all regrets are zero. The final Deadline-head target cost is

$$
c_t^{D,H}=c_{t,\mathrm{wf}}^D+c_{t,\mathrm{ep}}^D.
$$

### 14.1 Scale-free queue-weight calibration

The configured quantity is the desired queue share $s_q=0.20$, rather than a testbed-specific weight. During episodes 151--200, training accumulates

$$
I=\sum_t i_t,
\qquad
Q=\sum_t u_t.
$$

It then solves and freezes

$$
\beta_q
=\frac{s_q}{1-s_q}\frac{I}{Q}.
$$

The completed formal runs obtained

$$
\beta_q^{\mathrm{Small}}=5.604848,
\qquad
\beta_q^{\mathrm{Medium}}=4.355573,
$$

both realizing a queue share of 0.20.

## 15. Upper reliability cost

The terminal upper reliability cost is defined in log-reliability space:

$$
c_w^{R,H}
=\left[
\log R_w^{\mathrm{req}}-\log R_w
\right]_+.
$$

It is placed on the last transition of workflow $w$; all earlier reliability-head costs are zero. This preserves additivity while avoiding products inside the value target.

## 16. Upper Lagrangian updates

### 16.1 ESP-aligned Deadline multiplier

Let $e_D^{(n)}$ be the binary episode failure signal above. Over a rolling window of 60 episodes,

$$
\bar e_D=\frac{1}{60}\sum_n e_D^{(n)}.
$$

The target budget is $b_D^H=0.10$, which corresponds to a training-distribution target of $\operatorname{ESP}@0.90\ge0.90$. After the 50-episode warm-up, the PI-style update is

$$
z_{n+1}^H
=\Pi_{[0,6]}
\left(
z_n^H+0.1(\bar e_D-b_D^H)
\right),
$$

$$
\lambda_{D,n+1}^H
=\Pi_{[0,6]}
\left(
z_{n+1}^H+3.0(\bar e_D-b_D^H)
\right).
$$

### 16.2 Upper reliability multiplier

The upper reliability multiplier is updated for each completed workflow:

$$
\lambda_R^H
\leftarrow
\Pi_{[0,100]}
\left(
\lambda_R^H+10^{-3}c_w^{R,H}
\right).
$$

## 17. Lower-level state and action

The lower SAC actor receives a 9-dimensional local state conditioned on the server selected by the upper policy:

$$
s_t^L=
\left[
\tanh\left(\frac{L_{wi}}{10^{10}}\right),
\frac{T_{wi,k}^{\mathrm{exe}}(1)}{D_w^{\mathrm{rel}}},
\frac{[\widehat S_{wi,k}-t]_+}{D_w^{\mathrm{rel}}},
\frac{D_w^{\mathrm{abs}}-t}{D_w^{\mathrm{rel}}},
\frac{B_{wi,k}^{\mathrm{safe}}}{D_w^{\mathrm{rel}}},
\rho_{wi,k}^{\mathrm{safe}},
\tanh\left(\frac{P_k^{\mathrm{ind}}}{100}\right),
\tanh\left(\frac{P_k^{\mathrm{dyn}}}{250}\right),
R_w^{\mathrm{req}}
\right].
$$

The actor samples a Gaussian variable and applies a tanh transformation:

$$
z_t=\mu_\phi(s_t^L)+\sigma_\phi(s_t^L)\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,1),
$$

$$
\rho_t^{\mathrm{raw}}
=0.2+0.8\frac{\tanh(z_t)+1}{2}.
$$

The environment executes

$$
\rho_t=\operatorname{clip}
\left(\rho_t^{\mathrm{raw}},\underline\rho_{k_t},1\right).
$$

## 18. Action-local DVFS reward components

For the already selected server, the code constructs a $\rho=1$ counterfactual at the same decision state. Let $E_t^1,F_t^1,R_t^1$ be its energy, finish time, and task reliability, and let $E_t^\rho,F_t^\rho,R_t^\rho$ be the executed values.

The energy-saving component is

$$
e_t^L=E_t^1-E_t^\rho.
$$

The extra execution delay is

$$
\Delta T_t^L=[F_t^\rho-F_t^1]_+.
$$

Let $C_{w,t-1}^{\mathrm{partial}}$ be the partial workflow makespan before the action. The slack-aware time cost is

$$
h_t^L
=\frac{\Delta T_t^L}
{\max\left\{
D_w^{\mathrm{rel}}-C_{w,t-1}^{\mathrm{partial}},
0.05D_w^{\mathrm{rel}}
\right\}}.
$$

For replay normalization, each component $x$ is transformed as

$$
\mathcal{Z}(x)
=\operatorname{clip}
\left(
\frac{x-\mu_x}{\max(\sigma_x,\sigma_x^{\min})},
-3,3
\right),
$$

where the statistics are computed from the current replay buffer.

The unconstrained lower reward is

$$
r_t^{L,0}
=\mathcal{Z}(e_t^L)-\mathcal{Z}(h_t^L).
$$

The configured team-energy term is zero.

## 19. Lower reliability responsibility

The lower policy is charged only for reliability damage introduced by reducing frequency on the selected server. Let $\widehat\ell_t(R)$ be the projected final log reliability obtained from already scheduled tasks, candidate reliability $R$ for the current task, and a reliability estimate of 0.999 for every unscheduled task. Define

$$
v_t(R)
=\frac{
\left[
\log R_w^{\mathrm{req}}-\widehat\ell_t(R)
\right]_+
}{-\log R_w^{\mathrm{req}}}.
$$

The dense lower reliability cost is

$$
c_t^{R,L}
=\left[v_t(R_t^\rho)-v_t(R_t^1)\right]_+.
$$

Thus, a reliability deficit already present at $\rho=1$ remains an upper placement responsibility.

## 20. Workflow-level DVFS Deadline counterfactual

Local $\rho=1$ comparisons fail to capture queue propagation across workflows. The formal method therefore attributes Deadline damage by replaying the realized schedule under fixed placement and fixed server order.

### 20.1 Fixed schedule graph

Construct an auxiliary DAG $\mathcal{H}$ whose nodes are all executed tasks. It contains

1. every original workflow dependency edge, with its communication delay; and
2. an edge between every pair of consecutive tasks on the same realized server.

For a set of source workflows $\mathcal{U}$ restored to $\rho=1$, replay each node $v$ in topological order:

$$
\widetilde S_v^{(\mathcal{U})}
=\max\left\{
A_{w(v)},
t_v^{\mathrm{dec}},
\max_{u\in\operatorname{pred}_{\mathcal{H}}(v)}
\left(
\widetilde F_u^{(\mathcal{U})}+c_{uv}
\right)
\right\},
$$

$$
\widetilde F_v^{(\mathcal{U})}
=\widetilde S_v^{(\mathcal{U})}
+\widetilde T_v^{(\mathcal{U})},
$$

where $t_v^{\mathrm{dec}}$ is the realized decision time stored for node $v$, and

$$
\widetilde T_v^{(\mathcal{U})}
=
\begin{cases}
T_v(\rho_v)\rho_v=T_v(1), & w(v)\in\mathcal{U},\\
T_v(\rho_v), & \text{otherwise}.
\end{cases}
$$

### 20.2 Victim-level DVFS cost

Define the actual normalized Deadline violation of workflow $v$ as

$$
\ell_v
=\left[
\frac{C_v-D_v^{\mathrm{rel}}}{D_v^{\mathrm{rel}}}
\right]_+.
$$

Let $\widetilde\ell_v^{(\mathrm{all})}$ be the violation when all workflows are restored to $\rho=1$. The total Deadline damage attributable to DVFS for victim workflow $v$ is

$$
d_v^L
=\operatorname{clip}
\left(
\ell_v-\widetilde\ell_v^{(\mathrm{all})},0,1
\right).
$$

### 20.3 Source-to-victim marginal responsibility

Restore only source workflow $u$ to $\rho=1$ and let $\widetilde\ell_v^{(u)}$ be victim $v$'s resulting violation. Its marginal is

$$
m_{u\rightarrow v}
=\left[\ell_v-\widetilde\ell_v^{(u)}\right]_+.
$$

The victim cost is conserved and allocated among sources:

$$
a_{u\rightarrow v}
=d_v^L
\frac{m_{u\rightarrow v}}
{\sum_j m_{j\rightarrow v}}.
$$

If no individual restoration has a positive marginal, the allocation falls back to each source workflow's accumulated DVFS slowdown mass. The source-level replay credit is

$$
c_u^{D,L}=\sum_v a_{u\rightarrow v}.
$$

Finally, $c_u^{D,L}$ is distributed over the transitions of source workflow $u$ in proportion to their local time increases $\Delta T_t^L$:

$$
c_t^{D,L}
=c_u^{D,L}
\frac{\Delta T_t^L}
{\sum_{\tau\in\mathcal{T}_u}\Delta T_\tau^L}.
$$

This construction assigns both self-inflicted delay and cross-workflow queue externalities without double-counting victim cost.

## 21. Constrained lower SAC objective

The reward used by SAC is

$$
r_t^L
=r_t^{L,0}
-\lambda_D^L c_t^{D,L}
-\lambda_R^L c_t^{R,L}.
$$

With twin critics, the target is

$$
y_t
=r_t^L
+\gamma_L(1-d_t)
\left[
\min_{j\in\{1,2\}}
\bar Q_{\theta_j}(s_{t+1}^L,\rho_{t+1})
-\alpha\log\pi_\phi(\rho_{t+1}\mid s_{t+1}^L)
\right].
$$

The critic loss is

$$
J_Q(\theta_j)
=\mathbb{E}
\left[
\left(Q_{\theta_j}(s_t^L,\rho_t)-y_t\right)^2
\right],
$$

and the actor loss is

$$
J_\pi(\phi)
=\mathbb{E}
\left[
\alpha\log\pi_\phi(\rho_t\mid s_t^L)
-\min_j Q_{\theta_j}(s_t^L,\rho_t)
\right].
$$

The entropy temperature is fixed at $\alpha=0.2$ in the formal configurations.

## 22. Lower Lagrangian updates

### 22.1 Deadline multiplier

For the lower dual, an episode-level cost is the mean victim DVFS cost:

$$
\bar d_e^L=\frac{1}{N_e}\sum_v d_v^L.
$$

Over a rolling 40-episode window, the controller computes the mean of the worst 25%:

$$
\operatorname{CVaR}_{0.25}
=\frac{1}{|\mathcal{T}_{0.25}|}
\sum_{e\in\mathcal{T}_{0.25}}\bar d_e^L.
$$

With budget $b_D^L=0.08$, warm-up 20, learning rate 0.2, proportional gain 25, and maximum 8,

$$
z_{e+1}^L
=\Pi_{[0,8]}
\left(
z_e^L+0.2
\left(\operatorname{CVaR}_{0.25}-0.08\right)
\right),
$$

$$
\lambda_{D,e+1}^L
=\Pi_{[0,8]}
\left(
z_{e+1}^L+25
\left(\operatorname{CVaR}_{0.25}-0.08\right)
\right).
$$

### 22.2 Reliability multiplier

The lower reliability dual observes the terminal binary indicator

$$
b_w^R=\mathbb{I}(R_w<R_w^{\mathrm{req}}).
$$

Let $\bar b_R$ be the mean over the most recent 100 workflows. With budget $b_R^L=0.005$,

$$
\lambda_R^L
\leftarrow
\Pi_{[0,50]}
\left(
\lambda_R^L+0.5(\bar b_R-0.005)
\right).
$$

The dense per-transition reliability cost supplies the SAC learning signal, while the binary workflow indicator updates the multiplier against the actual reliability success criterion.

## 23. Training and target-network updates

The formal networks use

- upper D3QN: two hidden layers of 256 units, learning rate $10^{-4}$, batch size 256, five replay updates per episode;
- lower SAC: two hidden layers of 128 units, learning rate $3\times10^{-4}$, batch size 128, ten replay updates per episode;
- discount factors $\gamma_E=\gamma_D=\gamma_R=\gamma_L=0.99$;
- Polyak coefficient $\tau=0.005$;
- replay capacity 50,000 for both levels;
- gradient-norm clipping at 1.0;
- upper exploration $\epsilon:1.0\rightarrow0.05$ with decrement $5\times10^{-4}$;
- upper Deadline-positive replay fraction 0.30.

Both levels train jointly at episode end. Their Deadline multipliers are independent because placement and DVFS cause different forms of Deadline damage. The lower reliability multiplier is also independent of the upper reliability head.

## 24. Checkpoint selection

Every 20 training episodes, the deterministic policy is evaluated on two disjoint 20-seed validation blocks. For block $b$, define

$$
\operatorname{ESP}_{b}@0.90
=\frac{1}{20}
\sum_{s=1}^{20}
\mathbb{I}(\operatorname{DSR}_{b,s}\ge0.90).
$$

A checkpoint is feasible only if

$$
\min_b\operatorname{ESP}_{b}@0.90=1.0
$$

and the pooled reliability success rate is

$$
\operatorname{RSR}=1.0.
$$

Among feasible checkpoints, selection minimizes mean total computing energy. Reliability, ESP, and response time are used only as later tie-breakers. This feasibility-first rule matches the constrained objective: energy is compared only after both service constraints are satisfied.

## 25. Code-to-equation traceability

| Mathematical component | Primary implementation |
|---|---|
| DAG, tasks, dependency data | `src/simulator/workflow.py` |
| Poisson arrivals | `src/datasets/arrival_process.py` |
| Pegasus sampling | `src/datasets/pegasus_workflow_generator.py` |
| HEFT-based Deadline | `src/simulator/deadline.py`, `src/schedulers/heft.py` |
| Task timing and queues | `src/simulator/edge_cloud_env.py` |
| DVFS time, power, energy floor | `src/simulator/resource.py` |
| Frequency-dependent reliability | `src/simulator/reliability.py` |
| Upper state | `src/D3QN/state.py` |
| Three-head D3QN | `src/D3QN/reliability_aware_high.py` |
| Upper Deadline credit | `src/D3QN/dual_head_constrained.py` |
| Queue-weight calibration | `src/D3QN/queue_weight_calibrator.py` |
| Upper Deadline dual | `src/D3QN/deadline_lagrangian.py` |
| Lower state and SAC | `src/sac/hierarchical_low_sac.py` |
| Lower reward components | `src/sac/reward.py` |
| Lower Deadline/reliability duals | `src/sac/deadline_lagrangian.py`, `src/sac/reliability_lagrangian.py` |
| Workflow counterfactual credit | `tools/train_hierarchical_d3qn.py` |
| Formal trainer wiring | `tools/train_coupled_hierarchical_d3qn_sac.py` |
| Checkpoint ranking | `tools/hierarchical_common.py` |

## 26. Claims that must remain code-faithful in the paper

1. **The formal method uses a three-head upper D3QN.** The heads estimate energy reward, Deadline cost, and reliability cost separately; the multipliers combine them only for action selection.
2. **The lower action is continuous.** SAC selects one normalized DVFS ratio for the server selected by the upper policy.
3. **The critical-path safety quantity is a feature, not a hard shield.** The executed action is clipped to the energy-efficient floor, not to $\rho^{\mathrm{safe}}$.
4. **The formal runs are primary-only.** Backup duplication exists in the simulator but is disabled in both formal configurations.
5. **Reliability remains frequency dependent.** Disabling backup does not remove the reliability constraint.
6. **The optimized energy is task computing energy.** Transmission energy and always-on idle leakage are outside the current objective.
7. **Task sub-Deadlines are not active in the formal configurations.** Workflow Deadlines, remaining critical-path estimates, queue credit, and episode-level ESP credit are active.
8. **The upper reward includes realized lower-level energy.** This is the active `ep740_lagrangian_v1` cooperative profile.
9. **The lower Deadline responsibility is workflow-level and counterfactual.** It replays fixed placement and server order so that DVFS-induced queue externalities can be assigned across workflows.
10. **Checkpoint feasibility is stricter than mean DSR.** Both validation blocks must achieve `ESP@0.90 = 1.0`, and reliability success must equal 1.0 before energy is ranked.
11. **The optional wider DVFS-aware upper encoder is disabled.** Formal runs use the 12-feature-per-server state described above because `agent.dvfs_aware_high_state=false`.
12. **The optional physically scaled lower reward is disabled.** Formal runs use replay z-score normalization because `agent.physical_low_reward=false`.
