# The Benchmark Gap in Continual RL: From Continual World to SPIRAL
Article 1 of 12 | [CL][RL] | Anchor papers: Abel et al. 2023 · Khetarpal et al. 2022 · SPIRAL 2025 · AgarCL 2025 | Series: Continual Intelligence
---

We have had benchmarks for continual reinforcement learning for five years. They all measure the same wrong thing. Here is the proof — and a map of what we should be measuring instead.

---

## §1 — Why Benchmarks Define a Field

A field's benchmarks are not neutral measurement tools. They encode what the field believes the problem *is*. ImageNet told computer vision researchers that the problem was recognizing objects in labeled photographs. CIFAR told small-image representation learners that they needed a fast, accessible way to measure feature quality across ten classes. Those choices shaped hiring, funding, and publication norms for a decade each. The benchmark does not merely test a solution — it defines the solution space.

So what did continual reinforcement learning benchmarks define?

The field's dominant benchmark for four years was Continual World (Wolczyk et al., NeurIPS 2021): a suite of robotic manipulation tasks built on the Meta-World simulation environment. Continual World is well-constructed. It is computationally accessible. It exposed RL-specific challenges that supervised continual learning simply doesn't have, including sparse rewards, credit assignment across task switches, and the compounding effects of policy non-stationarity. The field owed it real progress.

But Continual World made a structural assumption so quietly embedded that most papers using it never acknowledged it: *tasks arrive with explicit boundary signals, and agents operate in episodic environments with resets between tasks*. The real world provides neither. A robot in continuous deployment does not receive a flag saying "Task 4 has started." A trading algorithm does not get reset to a neutral state when market conditions shift. An autonomous vehicle does not receive a checkpoint notification when it crosses from city driving to highway driving.

The benchmark defined the problem as task-switching capacity in episodic environments. The actual problem is non-episodic adaptation to stochastically non-stationary dynamics. These are not the same problem.

The gap between them has now been formally characterized. Khetarpal et al. (2022, JAIR) drew the critical distinction between **scope non-stationarity** (the distribution of tasks changes) and **driver non-stationarity** (the mechanism generating the non-stationarity itself changes). Current benchmarks test the former almost exclusively. The latter — the regime where the agent cannot characterize what is changing, only that something is — has been almost completely unmeasured.

To audit this gap precisely, we need a yardstick. Abel et al. (2023) gave us one.

> **Key Takeaway:** The benchmark is not merely a test — it encodes a theory of the problem. Four years of CRL benchmarks encoded a theory that real-world non-stationarity looks like episodic task-switching. It does not.

---

## §2 — The Formal Definition of Continual RL

Abel et al. (2023) provided the first rigorous axiomatic definition of continual reinforcement learning, and using it as a checklist reveals that traditional multi-task RL is a *degenerate special case* of the real problem.

The core commitment in Abel's formulation is the **implicit indefinite search process**: a continual RL agent is one that never stops learning. Not "keeps a replay buffer from old tasks." Not "performs well on a fixed task sequence." Literally never reaches a point where it should stop updating. The agent's policy π is always potentially improvable, because the environment distribution ℰ is never fully characterized.

This is not a technical nicety. It is a fundamental axiom that invalidates most current evaluation protocols. If your benchmark has a training phase and an evaluation phase — if there is a moment where you *stop the agent and measure it* — you have implicitly assumed that learning has a completion point. Abel's formulation rejects this assumption entirely.

From this axiom and Khetarpal et al.'s desiderata, five necessary properties emerge for a valid CRL benchmark:

---

> **The Five Necessary Criteria for a Valid CRL Benchmark**
>
> 1. **No task labels or boundary signals** — the agent is never told when tasks change
> 2. **No episodic resets between tasks** — the environment does not return to a neutral state at task boundaries
> 3. **No access to replay buffers from prior tasks** — the agent cannot revisit data from previous experience streams
> 4. **Non-stationarity is stochastic, not adversarial** — changes in the environment distribution are drawn from a stochastic process, not chosen by an adversary trying to minimize performance
> 5. **Performance measured as infinite-horizon average reward** — not peak performance on individual tasks, but the *rate* of reward accumulation across the agent's lifetime

---

The mathematical commitment in criterion (5) is the most consequential. Infinite-horizon average reward is the correct objective because it directly penalizes time spent not learning. A benchmark that measures per-task performance peak implicitly rewards an agent that learns task N quickly even if it destroys the features useful for tasks N+1 through N+k. The correct metric is ∑ r_t / T as T → ∞, not max_τ V^π_τ.

This framing also makes Abel's definition incompatible with traditional multi-task RL. In multi-task RL, the agent is allowed to stop learning — it converges to a policy that simultaneously performs all tasks, and convergence is success. In CRL, convergence is failure, because the environment is guaranteed to change. The best CRL agents are those that "carry out an implicit search process indefinitely" (Abel et al., 2023) — not those that arrive at a stable solution.

The remainder of this article uses these five criteria as the audit framework.

[→ A3: The Big World Hypothesis explains why this definition is mathematically necessary: any benchmark assuming a finite task set makes an assumption the real world does not satisfy.]

> **Key Takeaway:** Abel's formal definition reveals that most CRL evaluation assumes what it should prohibit — a completion point for learning. Continual RL requires indefinite search, not convergence.

---

## §3 — The Audit Table: No Benchmark Passes All Five Criteria

This is the central empirical claim of this article: no currently deployed CRL benchmark satisfies all five criteria. The audit table below makes this specific.

---

**Figure 1 — CRL Benchmark Audit Against the Five Necessary Criteria**

| Benchmark | (1) No Task Labels | (2) No Episodic Reset | (3) No Replay Access | (4) Stochastic Non-Stat. | (5) ∞-Horizon Metric | **Score** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Continual World (2021) | ⚠️ | ❌ | ❌ | ❌ | ⚠️ | **2 / 5** |
| Continuous Coordination (2021) | ✅ | ❌ | ❌ | ❌ | ❌ | **1 / 5** |
| AgarCL (2025) | ✅ | ✅ | ⚠️ | ✅ | ⚠️ | **4 / 5** |
| SPIRAL (2025) | ✅ | ✅ | ✅ | ✅ | ⚠️ | **4½ / 5** |
| MetaWorld (non-CRL baseline) | ❌ | ❌ | ❌ | ❌ | ❌ | **0 / 5** |

```
  Legend
  ──────────────────────────────────────────────────────────────────
  ✅  Criterion fully satisfied
  ⚠️  Criterion partially satisfied or implicitly present
  ❌  Criterion not satisfied

  Critical failure mode (2021–2024): No benchmark removes the
  episodic reset assumption — the single property the real world
  never provides. AgarCL (2025) is the first to close this gap.
  ──────────────────────────────────────────────────────────────────
```

*Figure 1: Audit of CRL benchmarks against the five necessary criteria. No benchmark before 2025 satisfied more than 1 of 5. The most critical failure: no pre-2025 benchmark removes the episodic reset. AgarCL and SPIRAL approach the ideal from different directions; neither closes all five.*

---

Walk through each row.

**Continual World** (Wolczyk et al., NeurIPS 2021): The benchmark provides a sequence of ten robotic manipulation tasks (HAMMER-V1, PUSH-WALL-V1, FAUCET-CLOSE-V1, PUSH-BACK-V1, STICK-PULL-V1, etc.) from the Meta-World suite. It earns a ⚠️ on criterion (1) because task identity is not explicitly given — but the agent can infer task boundaries from the episode structure, since each new task begins a fresh episode. It fails criterion (2) entirely: Meta-World's episode structure is preserved. Every task begins with a reset to an initial state. The agent always knows, structurally, that a task boundary has occurred. It fails criterion (3) because replay buffers are standard in off-policy baselines. It fails criterion (4) because the task sequence is fixed — the agent sees the same ten tasks in the same order; there is no stochastic process governing what comes next. On criterion (5), Continual World reports per-task performance averaged over the sequence; this is approximately useful but theoretically incorrect for CRL, earning a ⚠️.

The honest verdict on Continual World: it tests catastrophic forgetting prevention in a task-switching protocol. That is a real problem. But it is not the CRL problem as formally defined.

**Continuous Coordination** (Nekoei et al., 2021): A multi-agent cooperative scenario where two agents must coordinate on tasks that change. It satisfies criterion (1) — agents don't receive explicit task labels — which is its genuine innovation. But it fails on every structural criterion. The coordination tasks arrive episodically. There is no stochastic non-stationarity generating the task sequence. Performance is measured per task. The benchmark tests adaptation to task-switching in a cooperative setting; it does not test continual learning under genuine non-stationarity.

**AgarCL** (Tang et al., 2025): Finally, real progress. AgarCL builds on Agar.io — the browser game where cells grow by consuming smaller cells, split and merge, and compete in a continuously evolving ecosystem. The key properties that make it genuinely CRL-relevant: non-episodic dynamics (the game never resets; cells die but the ecosystem continues), stochastic ever-evolving non-stationarity (other players change strategy, new cells enter, the competitive landscape shifts continuously), continuous actions, and partial observability. This benchmark satisfies the two criteria that were completely unaddressed before 2025: no episodic reset and stochastic non-stationarity. The ⚠️ on criterion (3) reflects that replay access constraints are not explicitly enforced; the ⚠️ on criterion (5) reflects that performance metrics are still task-performance-oriented rather than lifetime average reward.

**SPIRAL** (Liu et al., 2025): SPIRAL takes a different path. It uses zero-sum self-play — agents competing against each other — as an implicit curriculum generator. In this protocol, there are no external task labels (the "tasks" are the game states generated by the current opponent), no episodic resets in the usual sense (self-play is continuous), and no replay of old opponent policies (the curriculum is endogenous). The stochastic non-stationarity criterion is satisfied by construction: as the agent improves, the opponent improves, and the distribution of game states changes. The ⚠️ on criterion (5) reflects that SPIRAL's metrics are downstream reasoning benchmarks (math accuracy, general reasoning) rather than CRL-native infinite-horizon reward signals. This is a domain limitation, not a fundamental failure.

The key empirical anchor from SPIRAL: training Qwen3-4B-Base on Kuhn Poker — a simple card game — with no mathematical supervision yields 8.6% improvement on math benchmarks and 8.4% improvement on general reasoning. The model received no math training. The improvement came entirely from competitive self-play dynamics that forced systematic decomposition, expected value calculation, and case-by-case analysis. This is the closest existing result to demonstrating what a genuine CRL evaluation protocol would reveal: endogenous curriculum generation driving general capability improvement.

> **Key Takeaway:** The audit reveals a four-year gap in the benchmark landscape. Every CRL benchmark before 2025 failed on the two most critical criteria: episodic resets and stochastic non-stationarity. AgarCL and SPIRAL close different parts of this gap — neither closes all of it.

---

## §4 — Continual World: The Current Standard and Its Ceiling

Continual World deserves honest analysis, not dismissal. Wolczyk et al. (NeurIPS 2021) built something genuinely useful, and understanding precisely where it hits its ceiling tells us what the next generation of benchmarks needs.

The genuine contributions of Continual World: it provided the field with a computationally accessible, diverse set of tasks that exposed RL-specific challenges absent from supervised continual learning. When a network trained on task 1 (pushing a button) then trains on task 2 (opening a door), the policy interference is qualitatively different from supervised continual learning — the reward signal changes, the state distribution shifts, and the effect of forgetting is compounded through policy non-stationarity. Earlier CRL work had mostly tested on variants of Atari, which share similar visual inputs and action spaces across games. Continual World's robotic manipulation tasks are genuinely diverse in their kinematic demands.

It also, critically, highlighted forward transfer as an underinvestigated problem. The Continual World paper itself flags this: most methods that prevent catastrophic forgetting do not improve performance on *future* tasks; they merely protect past performance. Forward transfer — the degree to which learning task 1 accelerates learning task 2 — was flagged as the more interesting and less studied phenomenon.

That flag was largely ignored. The papers that have used Continual World since 2021 overwhelmingly optimize for backward transfer (forgetting prevention) rather than forward transfer. This is the benchmark's most damaging side effect: by providing an easy forgetting-measurement protocol, it channeled research toward forgetting metrics while the forward-transfer problem accumulated.

The structural failure is the episodic reset. Meta-World, the simulation environment on which Continual World is built, operates in episodes with reset states. When Continual World cycles through tasks, each new task begins with a fresh episode. The agent's optimizer state may carry over, but the environment state does not. This means the agent structurally knows when task switches occur — a new episode beginning after a period of training on different tasks is a boundary signal, even if it is never labeled as one.

Real-world deployment doesn't work this way. A robotic arm in a warehouse doesn't return to a home position when assigned a new task type. A recommendation system doesn't receive a reset signal when user preferences gradually shift. The episodic structure of Continual World is not a minor limitation — it is the mechanism that makes the benchmark tractable while also making it an imperfect proxy for the actual problem.

---

**Figure 2 — CRL Benchmark Timeline: Gaps Opened and Closed**

```
                       CRL BENCHMARK TIMELINE
  ════════════════════════════════════════════════════════════════════════

          2021                       2023                    2025
            │                          │                  │        │
  ──────────●──────────────────────────●──────────────────●────────●──────▶
            │                          │                  │        │
      ╔═════╧══════════════╗     ╔═════╧═════╗      ╔════╧═══╗ ╔══╧═════╗
      ║  CONTINUAL WORLD   ║     ║   ABEL    ║      ║ AgarCL ║ ║ SPIRAL ║
      ║  NeurIPS 2021      ║     ║  FORMAL   ║      ║  2025  ║ ║  2025  ║
      ║                    ║     ║   DEF.    ║      ╚════════╝ ╚════════╝
      ║  + CONTINUOUS      ║     ║ NeurIPS   ║
      ║  COORDINATION      ║     ║   2023    ║
      ╚════════════════════╝     ╚═══════════╝

  ────────────────────────────────────────────────────────────────────────
  CLOSES          Task diversity gap vs. Atari; RL-specific challenges
                  (sparse rewards, policy non-stationarity) identified

  LEAVES OPEN     ✗ Episodic reset (every task begins with env reset)
                  ✗ Stochastic non-stationarity (fixed task sequences)
                  ✗ Forward transfer (flagged but not measured)
  ────────────────────────────────────────────────────────────────────────
  CLOSES          Field's conceptual gap codified; "implicit indefinite
                  search" formalized; five desiderata specified

  LEAVES OPEN     Benchmark infrastructure gap (unfilled — no new suite)
  ────────────────────────────────────────────────────────────────────────
  CLOSES          ✓ Non-episodic dynamics (Agar.io ecosystem never resets)
  (AgarCL)        ✓ Stochastic non-stationarity (endogenous via competitors)
                  ✓ Continuous actions + partial observability

  LEAVES OPEN     Replay constraint implicit only; metrics task-performance
  ────────────────────────────────────────────────────────────────────────
  CLOSES          ✓ Endogenous infinite curriculum (self-play)
  (SPIRAL)        ✓ No task labels; continuous competition
                  ✓ Unlimited non-stationarity via opponent adaptation

  LEAVES OPEN     CRL-native infinite-horizon metrics not yet defined
  ════════════════════════════════════════════════════════════════════════
  The episodic-reset gap persisted across ALL benchmarks from 2021–2024.
  AgarCL (2025) is the first to close it.
```

*Figure 2: Timeline of CRL benchmark releases against the gap each was designed to close and the gap it left open. The non-stationarity-without-reset gap persisted for four years. AgarCL closes it through environmental non-stationarity; SPIRAL approaches it through competitive self-play.*

---

## §5 — AgarCL: What a Real CRL Environment Looks Like

AgarCL (Tang, Obando-Ceron et al., 2025) is the first benchmark that satisfies both the non-episodic and stochastic non-stationarity criteria simultaneously. It is also the hardest benchmark currently available. That is not a coincidence.

The underlying game, Agar.io, was designed as a competitive browser game, not as a research environment. Cells grow by consuming smaller cells. Players can split their cell mass, sacrificing mobility for area coverage. Cells can eject mass to propel themselves or to bait opponents. The competitive ecosystem is continuously evolving: as each player's mass changes, the relationship between players changes, which changes optimal strategy, which changes the behavior of other players, which changes the state distribution the agent encounters. There is no natural episode boundary. The game's non-stationarity is endogenous to its competitive dynamics.

The researchers' key insight: this makes Agar.io structurally honest about CRL. The difficulty is not designed — it emerges from the interaction between agents in a non-stationary environment. A standard episodic benchmark can be "solved" by memorizing a good policy for each task; AgarCL cannot, because the tasks themselves change as a function of the agent's behavior.

The specific properties that satisfy the CRL criteria:

**Non-episodic dynamics**: Cells in the game live and die without resetting the global state. When a cell is consumed, the consumer grows; the map continues. The learning agent cannot assume a clean slate between objectives.

**Stochastic ever-evolving dynamics**: Other players adapt. The distribution of opponent behaviors shifts over the course of training in ways that cannot be predicted from the initial conditions. This is the operationalization of Khetarpal's "driver non-stationarity" — not just the distribution of tasks changing, but the mechanism generating the distribution changing.

**Continuous actions and partial observability**: The agent controls cell movement with continuous force vectors and can only observe cells within a limited field of view. These are not arbitrary design choices — they are properties that episodic benchmarks routinely circumvent by providing full state observations and discretizing the action space.

The baseline results are instructive: PPO, DQN, and SAC all fail to achieve consistent improvement in AgarCL over extended training horizons. This should be celebrated, not treated as a problem to be fixed. A benchmark where all standard baselines fail equally tells you that you have found a genuine unsolved problem. A benchmark where one carefully tuned baseline succeeds at 70% tells you that the field has found one particular solution; it doesn't tell you the shape of the remaining problem.

"The few simulators explicitly designed for empirical research in continual RL are often limited in scope or complexity" — this is the motivation the AgarCL authors state directly. Agar.io's complexity was not added to increase difficulty; it was inherited from a system where complexity serves a function (competitive engagement). That functional complexity is exactly what CRL research needs.

> **Key Takeaway:** AgarCL is hard because real non-stationarity is hard. The failure of standard baselines is a feature. A benchmark that reveals the full scope of the problem is more valuable than one that permits incremental progress on a simplified version of it.

---

## §6 — SPIRAL: Self-Play as an Implicit CRL Protocol

SPIRAL (Liu et al., 2025) solves the biggest benchmark design problem accidentally: it generates an infinite, endogenous curriculum where difficulty adapts continuously to the agent's current level. This is exactly what Abel's formal definition requires, arrived at through a completely different route.

The technical innovation in SPIRAL is role-conditioned advantage estimation (RAE). In multi-agent self-play, the naive approach of training each agent against a snapshot of the other agent's policy is notoriously unstable — the curriculum oscillates rather than progressing. RAE stabilizes multi-agent training by normalizing rewards relative to each player's expected performance — without it, models suffer from thinking collapse, where the policy degenerates to single repeated outputs. This is the mechanism that makes self-play curricula stable at scale.

The key result from SPIRAL deserves careful presentation. A Qwen3-4B-Base model trained exclusively on Kuhn Poker — a minimal card game with three cards, two players, and a small discrete action space — achieved 8.6% improvement on math benchmarks and 8.4% improvement on general reasoning benchmarks, relative to a model that received no game training. There was no mathematical content in the training data. The training signal was purely competitive: win more than your opponent at a card game.

Why does this work? The competitive pressure of zero-sum self-play forces the agent to develop systematic, domain-general reasoning capacities. Kuhn Poker requires: assessing probability distributions over hidden information (opponent's card), computing expected value of actions, case-by-case analysis of opponent strategies, and updating beliefs as the game progresses. These are the same cognitive operations that math problem-solving requires. The benchmark did not teach math — it developed the underlying reasoning capacities that transfer to math.

The implication for CRL benchmark design is significant: **competitive pressure can replace curriculum design**. Traditional benchmarks require a human designer to specify task sequences, difficulty progressions, and evaluation protocols. Self-play generates these endogenously. The curriculum is always calibrated to the current agent's ability because the agent's opponent *is* the current agent. This satisfies Abel's "implicit indefinite search" criterion by construction — there is no opponent that has been solved; improving the agent always generates a harder opponent.

SPIRAL's limitation as a CRL benchmark is its metrics. The evaluation is downstream: did math performance improve? This is useful for demonstrating general capability transfer, but it is not an infinite-horizon reward metric in the CRL sense. A true CRL evaluation of SPIRAL would measure the *rate* of improvement in game performance across the training lifetime, not performance on held-out benchmarks at the end. The benchmark is a proof of concept for the self-play protocol; the evaluation methodology is borrowed from the LLM fine-tuning literature and has not been redesigned for CRL's specific measurement needs.

> **Key Takeaway:** SPIRAL demonstrates that competitive self-play is a structurally valid CRL protocol — it generates an endogenous, infinitely deep curriculum that never terminates. The evaluation methodology lags the protocol design; aligning SPIRAL's metrics with CRL's infinite-horizon requirements is the natural next step.

---

## §7 — The Missing Criteria: A Research Agenda

Three desiderata from Khetarpal et al. (2022) remain currently unmeasurable — not because researchers have ignored them, but because no benchmark provides the infrastructure to measure them.

---

**Figure 3 — CRL Benchmark Coverage Map**

```
  NON-STATIONARITY RICHNESS
  (driver non-stat · stochastic dynamics · no boundary signals)

    HIGH  │                         ╔══════════════════════════════╗
          │                         ║     ★   IDEAL REGION        ║
          │                         ║                              ║
          │        SPIRAL ◆         ║        AgarCL ●             ║
          │        (separate        ╚══════════════════════════════╝
          │         reasoning
          │         plane)                   ↑
          │                           first benchmark
  MED     │                          to reach this zone
          │
          │                  ● Continuous Coordination
          │                    (multi-agent, still episodic)
    LOW   │  ● MetaWorld            ● Continual World
          │  (non-CRL)               (the field standard, 2021–2025)
          │
          └───────────────────────────────────────────────────────▶
               NARROW                                       BROAD
                              TASK DIVERSITY

  ─────────────────────────────────────────────────────────────────
   DEAD ZONE (high diversity, no stochastic non-stat): most current
   benchmarks cluster here. Diverse tasks; static episodic sequences.

   ◆ SPIRAL occupies a separate evaluation plane (downstream reasoning
     benchmarks) and is plotted here for orientation only.
  ─────────────────────────────────────────────────────────────────
```

*Figure 3: Coverage map of CRL benchmarks across task diversity and non-stationarity richness. The majority cluster in the bottom half — broad task sets but static, episodic sequences. The CRL agent must operate in the top-right quadrant. AgarCL is the first to place a dot there.*

---

The three missing criteria are not exotic research goals. They are direct consequences of the formal definition.

**Missing Criterion 1: Performance Under Task Uncertainty**

No current benchmark explicitly measures decision quality under genuine non-stationarity without boundary signals. This is distinct from forgetting. Forgetting is measured by asking: how well does the agent perform on task τ after training on tasks τ+1 through τ+k? Task uncertainty is measured by asking: how well does the agent perform when it cannot tell whether a task switch has occurred, and must simultaneously maintain good performance on the current task while remaining ready to switch?

The distinction matters for algorithm design. Algorithms optimized for forgetting prevention are not the same as algorithms optimized for task uncertainty management. The former assume they know when the regime changes; the latter cannot. No benchmark separates these failure modes.

**Missing Criterion 2: Sample Efficiency Across the Lifespan**

Standard CRL benchmarks report per-task asymptotic performance. They do not report the *rate* of learning — how quickly the agent reaches its performance ceiling on each successive task. This matters because a CRL agent that reaches 80% performance on each new task in 1,000 steps is qualitatively better than one that reaches 80% performance in 100,000 steps, even if their final performance is identical. The former has more total reward across its lifetime; the latter spends more time in the slow-learning phase.

The infinite-horizon average reward criterion (criterion 5 in the audit table) captures this if implemented correctly. None of the current benchmarks implement it correctly. They report peak performance, not reward rate.

**Missing Criterion 3: Compositional Transfer**

Can an agent that has learned task A and task B acquire task C faster, where C requires sub-skills present in both A and B but not identical to either? This is compositional transfer, and it is the core of what forward transfer should measure.

Continual World's own authors flagged forward transfer as under-studied. The gap has not been filled. Measuring compositional transfer requires tasks that share sub-structure — carefully designed so that the shared components are known to the experimenter and measurable in the agent. Building such a task suite for RL is harder than for supervised learning because sub-skill identification in RL policies is itself unsolved. But the measurement gap is not an excuse to ignore the problem; it is the call to work on it.

> **Key Takeaway:** The three unmeasurable criteria — task uncertainty management, lifetime sample efficiency, and compositional transfer — are not peripheral concerns. They are what distinguishes a genuine continual learner from a sophisticated multi-task RL agent. The benchmark infrastructure to measure them does not exist yet.

---

## §8 — CL as Computationally Constrained RL: The Holistic Objective

Kumar et al. (2025) provide the only rigorous theoretical treatment of what a CRL benchmark should actually optimize for. Their formulation: continual learning is **infinite-horizon average reward maximization under a compute constraint**.

This single sentence reorganizes several confused debates in the field.

The compute constraint is what distinguishes CRL from RL. An unconstrained RL agent can, in principle, maintain a replay buffer of all past experience and retrain from scratch when the environment changes. A CRL agent cannot — it operates under a fixed computational budget per timestep. Forgetting is not a failure of memory; it is an inevitable consequence of the compute constraint, and should be analyzed as such.

The reformulation changes what counts as good forgetting. Under the per-task performance metric standard in Continual World, forgetting task A while learning task B is always costly — the agent's performance on A decreases, decreasing the average. Under the infinite-horizon average reward metric, forgetting information about task A is *correct* if task A does not recur. Maintaining detailed representations of a non-recurring task wastes capacity that could be used for future tasks. The Kumar formulation makes this precise: "forgetting non-recurring information is not catastrophic by this definition."

This reframes the Continual World evaluation entirely. The recommended final-evaluation variant (CW20) runs the same 10 tasks twice in sequence — every task recurs exactly once. In this regime, forgetting any task is genuinely costly — the task will recur. But the protocol is circular: it assumes recurring tasks and then measures forgetting as bad. In the real world, many tasks do not recur. The benchmark artificially maximizes the cost of forgetting, which is why forgetting prevention dominates the literature.

The correct benchmark design, per Kumar et al., requires specifying a distribution over task recurrence — what fraction of tasks will be encountered again, and at what rate? Different distributions require qualitatively different agent strategies. No current benchmark makes this distribution explicit.

> **Key Takeaway:** Kumar et al. (2025) provide the theoretical ground for the benchmark audit: the correct objective is infinite-horizon average reward under a compute constraint. This makes forgetting-prevention the wrong primary metric, and makes task recurrence distribution the key missing variable in benchmark design.

---

## §9 — What Comes Next

**← This article established:** The field has been measuring the wrong things for four years. CRL benchmarks test task-switching capacity in episodic environments; real CRL requires non-episodic, stochastically non-stationary adaptation. Five criteria define a valid CRL benchmark. No current benchmark satisfies all five. AgarCL and SPIRAL are the closest existing approximations, from different directions.

**→ A1 (The Plasticity Crisis):** Now we understand *why* plasticity collapse is so difficult to detect in the literature — the benchmarks that would reveal it at full scale don't yet exist. Standard task sequences of 10–50 look fine; the collapse manifests at 500 or more sequential tasks, the scale that genuine non-episodic environments would produce. Article 1 shows what happens to networks at that scale, and why the failure is structural rather than incidental.

**→ A3 (The Big World Hypothesis):** Why the benchmark gap is mathematically inevitable. Any benchmark assuming a finite, known task set makes an assumption the real world does not satisfy. The world model hypothesis explains why CRL's indefinite search requirement is not a design choice but a mathematical consequence of operating in an open world.

The motivating question this article closes with: what would a CRL system that passes all five criteria actually look like? It would operate in a non-episodic environment with stochastic non-stationarity. It would never receive task boundary signals. It would be evaluated on lifetime reward rate, not per-task peak. It would need to manage task uncertainty, maintain sample efficiency across an unbounded lifespan, and achieve compositional transfer to tasks outside its training distribution.

Building that system is the subject of the next eleven articles.

---

> **Final Key Takeaways**
>
> 1. **Abel's formal definition makes most CRL benchmarks invalid.** The requirement of "never stop learning" is incompatible with episodic reset structures. Multi-task RL is a degenerate special case of CRL, not its foundation.
>
> 2. **Forward transfer is more important than catastrophic forgetting prevention** — and less studied, because existing benchmarks don't measure it at lifetime scale. The Continual World paper flagged this in 2021; four years later, the measurement gap remains.
>
> 3. **SPIRAL and AgarCL are the first benchmarks that approach a real CRL protocol** — one through competitive pressure generating an endogenous curriculum, one through environmental non-stationarity without episodic resets. Neither fully closes the gap; together they show where the gap can be closed.
>
> 4. **Kumar et al.'s formulation resolves the forgetting debate.** Under infinite-horizon average reward, forgetting non-recurring information is correct, not catastrophic. The benchmark is the problem; the agent is doing the right thing.
>
> 5. **Three criteria remain unmeasurable:** performance under task uncertainty, lifetime sample efficiency, and compositional transfer. These are not future research directions — they are the current measurement gap. Closing them is the concrete research agenda.

---

*Next: [A1 — The Plasticity Crisis in Continual Deep Learning →]*

*The benchmarks don't run agents long enough to see plasticity collapse. Here's what happens when they do.*

---

## References

[1] Abel, D., Barreto, A., Van Roy, B., Schölkopf, B., Silver, D., & Singh, S. (2023). **A definition of continual reinforcement learning.** *Advances in Neural Information Processing Systems (NeurIPS)*, 36. arXiv:2307.11046.

[2] Khetarpal, K., Riemer, M., Rish, I., & Precup, D. (2022). **Towards continual reinforcement learning: A review and perspectives.** *Journal of Artificial Intelligence Research (JAIR)*, 75, 1401–1476. arXiv:2012.13490.

[3] Wolczyk, M., Zając, M., Pascanu, R., Kuciński, Ł., & Miłoś, P. (2021). **Continual World: A robotic benchmark for continual reinforcement learning.** *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 28496–28510.

[4] Nekoei, H., Badrinaaraayanan, A., Courville, A., & Chandar, S. (2021). **Continuous coordination as a realistic scenario for lifelong learning.** *Proceedings of the International Conference on Machine Learning (ICML)*, 139, 8016–8024.

[5] Tang, Y., Obando-Ceron, J. S., et al. (2025). **AgarCL: The cell must go on — a benchmark for continual reinforcement learning.** arXiv:2505.18347.

[6] Liu, X., et al. (2025). **SPIRAL: Self-play on zero-sum games improves LLM reasoning via scalable curriculum.** arXiv:2506.24119.

[7] Kumar, S., et al. (2025). **Continual learning as computationally constrained reinforcement learning.** *Reinforcement Learning Journal*, 2025. arXiv:2307.04345.

[8] Yu, T., Quillen, D., He, Z., Julian, R., Hausman, K., Finn, C., & Levine, S. (2020). **Meta-World: A benchmark and evaluation for multi-task and meta reinforcement learning.** *Conference on Robot Learning (CoRL)*. arXiv:1910.10897. *(Referenced as the base environment for Continual World.)*
