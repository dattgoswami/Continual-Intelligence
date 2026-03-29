# Stable Deep RL at Scale: Gradients, KL, and the Shape of Learning

Article 8 of 12  | [RL][RE] | Anchor papers: Shah et al. arXiv:2512.21852 · Creus Castanyer et al. arXiv:2506.15544 · Saheb et al. arXiv:2602.19373 · Liu et al. arXiv:2510.01656 · Schaul et al. (2022) |
Series: Continual Intelligence

---

Pull up the loss curve from a standard RLHF run on a large language model. It looks fine — smoothly descending, plateauing, converging. Now pull up the KL divergence from the same run. What you see is not a curve. It is a seismograph. Spikes at irregular intervals, some reaching multiples of the smooth baseline, each one representing a moment when the policy's distribution lurched away from the reference and then — sometimes — lurched back. Shah, Obando-Ceron, Jain, Bartoldson, Kailkhura, Mittal, Berseth, Castro, Bengio, Malkin, Jain, and Venkatraman (2025) named this paper "A Comedy of Estimators" for exactly this reason: the standard KL regularization machinery in RL training of LLMs fails in ways that are, objectively, amusing to look at.

The loss curve lied. The policy was not converging. It was bouncing — a distribution doing chaotic work under a surface that numerical loss averaging was too blunt to resolve.

This article is a practitioner's toolkit. Each section is one diagnostic or fix. The structure is not a narrative arc; it is a checklist you can apply to your own training run. By the end, you will have five concrete stability checks that cost nothing to implement and have stopped training runs from failing silently. The previous article [← A5] established that the *shape* of reasoning traces — their distributional properties, not just their terminal answers — is the primary carrier of reasoning capability. What A5 showed for reasoning traces, A7 now shows for RL training itself: the shape of learning matters. Gradient distributions, representation geometry, and policy update dynamics collectively determine whether RL training builds capability or erodes it. Creus Castanyer, Obando-Ceron, Li, Bacon, Berseth, Courville, and Castro (2025) make the point directly: scaling deep RL networks is challenging and often results in degraded performance, yet the root causes of this failure mode remain poorly understood.

This article closes that gap.

---

## §1 — Why RL Stability at Scale Is a Distinct Problem

The conventional ML engineer's toolkit for training stability — gradient clipping, learning rate warmup, batch normalization — was developed for supervised learning on stationary data distributions. RL at scale is not supervised learning on stationary data. It is a fundamentally different stability problem with different failure modes, different diagnostic signals, and different fixes.

Three things change when you scale RL to LLM size that do not change in supervised learning:

**Non-stationarity compounds.** In supervised learning, the training distribution is fixed. In RL, the policy generates its own training data. As the policy changes, the data distribution changes, which changes the policy further. This feedback loop creates non-stationarity at every level: reward signals, value estimates, and representation targets are all moving simultaneously. Saheb, Obando-Ceron, Courville, Bashivan, and Castro (2026) characterize this as the primary driver of unstable training dynamics — not architecture choices, not hyperparameters, but the structural non-stationarity of the learning objective itself.

**Sparse rewards magnify small distribution shifts.** Supervised learning has a dense error signal: every training example provides a gradient. LLM reasoning tasks with binary correctness rewards provide a gradient only on correct or borderline trajectories. This sparsity means that small shifts in the policy distribution can have outsized effects — a policy that drifts slightly out of the reward-positive region sees its gradient signal collapse, which can trigger further drift. The critics that stabilize standard deep RL become computationally intractable at LLM scale, leaving RL pipelines with degraded stability infrastructure.

**Scale amplifies gradient anisotropy.** Deep RL networks at small scale have been extensively tuned. At LLM scale — hundreds of layers, billions of parameters, attention mechanisms not designed for RL gradient flow — gradient distributions develop pathological structure: some layers receive high-magnitude gradients throughout training while others receive near-zero gradients. These dead layers and exploding layers are not visible in the aggregate loss curve. They require per-layer monitoring to detect.

The Chandra, Agrawal, Hosseini et al. (2025) Shape of Thought paper [← A5], used here as the background framing, documents the downstream consequence: RL post-training changes the distributional shape of reasoning traces in ways that accuracy metrics do not capture. What looks like a stable training run — smoothly decreasing loss, improving accuracy — can be simultaneously collapsing the diversity and generalization capacity of the reasoning distribution. The stability failure is not the loss spike you see. It is the distribution change you don't.

(see Figure 1)

---

**Figure 1 — The Three RL Stability Failure Modes**

```
  ══════════════════════════════════════════════════════════════════════════════
   THREE FAILURE MODES SPECIFIC TO DEEP RL AT LLM SCALE
  ──────────────────────────────────────────────────────────────────────────────

  FAILURE MODE          WHAT BREAKS              VISIBLE IN?        COVERED IN
  ──────────────────────────────────────────────────────────────────────────────
  KL ESTIMATOR BIAS     Policy distribution      KL curve           §2
                        drifts without            (not loss)
                        reliable control signal

  GRADIENT ANISOTROPY   Dead / exploding layers  Per-layer grad     §3
                        invisible in loss curve   magnitude heatmap

  REPRESENTATION        Anisotropic embeddings   Embedding          §4
  COLLAPSE              → unstable value tracking singular values

  MISSING CRITIC        Sparse reward signal     Reward variance    §5
                        → high-variance policy    / policy entropy
                        updates

  UNBOUNDED REASONING   Exponential state space  Episode length     §6
  DEPTH                 growth → training         histogram
                        instability

  PARALLELISM BIAS      Biased data collection   Return variance    §7
                        → overfitting on         by env count
                        stale transitions

  POLICY CHURN          Greedy policy changes    Out-of-batch       §8
                        across out-of-batch       action divergence
                        states
  ──────────────────────────────────────────────────────────────────────────────
  [DIAGNOSTIC] ─────────────────────────────────► [PRESCRIPTIVE]
   KL curve, grad heatmap, embedding geometry → fixes in §2–§8
  ══════════════════════════════════════════════════════════════════════════════
```

*Figure 1: Seven stability failure modes specific to deep RL at scale. The left column is the failure mode; the right column is the minimal diagnostic signal required to detect it. Each row corresponds to one section of this article. The key insight is that most of these failures are invisible in the aggregate loss curve — they require purpose-built diagnostic signals to surface.*

---

> **Key Takeaway:** RL stability at LLM scale is a qualitatively different problem than supervised learning stability. Non-stationarity compounds, sparse rewards magnify distribution drift, and gradient anisotropy at depth creates hidden failure modes. The standard diagnostic — aggregate loss — is insufficient. Each failure mode has its own characteristic signal that must be monitored separately.

---

## §2 — A Comedy of KL Estimators

KL divergence from the reference policy is the central safety signal in RLHF and RL post-training for LLMs. It measures how far the current policy has drifted from the reference SFT model. Bounded KL means the policy is staying close enough to the reference that its outputs remain semantically coherent; exploding KL means the policy is departing from the reference's learned language distribution, often a precursor to reward hacking or mode collapse.

The problem is that the standard KL estimators used in practice are unreliable.

Shah, Obando-Ceron, Jain, Bartoldson, Kailkhura, Mittal, Berseth, Castro, Bengio, Malkin, Jain, and Venkatraman (2025) audit the KL estimation landscape in RL training of LLMs and find a comedy of failures: different estimators diverge from the true KL in different directions, producing regularization that is simultaneously over-penalizing in some regimes and under-penalizing in others. The divergence spikes characteristic of large-scale LLM RL training are, in significant part, an artifact of this estimator mismatch rather than evidence that the policy has genuinely escaped the reference distribution.

The practical consequence is bidirectional. First, the KL regularization term — intended to prevent policy collapse — may be applying pressure at the wrong times and in the wrong directions. A policy that is legitimately exploring a useful direction can be penalized back toward the reference by an estimator that misidentifies the exploration as a dangerous divergence. Second, genuine distribution collapse may be accompanied by an estimated KL that looks normal, because the estimator's bias happens to cancel the true divergence at that operating point. The comedy is that the safety mechanism is unreliable in both directions simultaneously.

**The diagnostic signal**: monitor both the estimated KL and an independent proxy for distribution divergence. Output token distribution shift, average log-probability under the reference model, and sentence-level embedding cosine similarity to the reference distribution provide cross-validation for the KL estimate. When estimated KL is flat but these proxies are moving, the estimator has failed in the dangerous direction. When estimated KL is spiking but proxies are stable, the estimator is over-penalizing.

**The prescriptive fix**: use multiple KL estimators in parallel and cross-check them. The divergence between estimators is itself informative: high inter-estimator variance indicates an operating regime where no single estimate can be trusted. In that regime, treat the maximum of all estimators as the working KL value — conservative, but resistant to the systematic under-estimation failure.

(see Figure 2)

---

**Figure 2 — The KL Comedy: Estimated vs. True Divergence**

```
  ══════════════════════════════════════════════════════════════════════════════
   KL DIVERGENCE DURING LLM RL TRAINING
   (After Shah et al., 2025)
  ──────────────────────────────────────────────────────────────────────────────

  KL
  Divergence
    ▲
    │                           *
  6 ┤                          ***          *
    │                         *****        ***
  4 ┤      *                 *******      *****
    │     ***   *           *       *    *     *
  2 ┤    *****_***___________         ____       ________
    │___/                                                  \_______
  0 ┤
    └────────────────────────────────────────────────────────────►
             Training Steps  →                        (arb. units)

  ─────────────────────────────────────────────────────────────────────────────
  ▓▓▓ = estimated KL (standard estimator)   ── = loss curve (reference)
  * = true KL divergence (cross-validated)

  KEY OBSERVATION: Spikes in the estimated KL curve do not always correspond
  to genuine policy divergence. Some represent estimator failure; others mask
  genuine but undetected divergence during the flat segments. The loss curve
  (bottom reference) is smooth throughout — standard monitoring would not
  surface these events.
  ══════════════════════════════════════════════════════════════════════════════
```

*Figure 2: The KL Comedy in schematic form. Spikes in the estimated KL divergence curve are a characteristic signature of large-scale LLM RL training. Shah et al. (2025) establish that these spikes reflect estimator failure as much as genuine policy divergence — the standard KL estimator is unreliable precisely when its signal is most needed. The loss curve (lower reference line) remains smooth throughout, demonstrating that standard training monitoring would miss these events entirely.*

---

> **Key Takeaway:** Standard KL estimators in LLM RL training are unreliable in both directions — they over-penalize genuine exploration and under-penalize genuine collapse. Shah et al. (2025) characterize this failure systematically. The fix: run multiple estimators in parallel, treat their disagreement as an instability signal, and cross-validate against independent distribution-shift proxies.

---

## §3 — Stable Gradients for Stable Learning

The second failure mode is deeper in the architecture: the gradient distribution itself becomes pathological at scale. Creus Castanyer, Obando-Ceron, Li, Bacon, Berseth, Courville, and Castro (2025) confront this directly. Their starting observation — that scaling deep RL networks is challenging and often results in degraded performance, yet the root causes of this failure mode remain poorly understood — captures the field's state before this work: practitioners knew scaling broke things, but not precisely where.

The answer is in the gradient distribution across layers.

At small network sizes, gradient magnitudes are roughly uniform across layers: each layer receives enough gradient signal to update its representations meaningfully. At large network sizes, gradient flow through deep RL networks becomes anisotropic — some layers receive gradients orders of magnitude larger than others. The large-gradient layers update rapidly, potentially overshooting their optimal configurations. The near-zero-gradient layers effectively stop learning, becoming frozen on whatever representations they held at the point when their gradient signal died. The network has layers that are functionally dead without any obvious sign in the output: no NaN gradients, no loss spikes, no visible training failure.

The aggregate loss curve does not detect this because a dead layer in the middle of a deep network does not immediately cause loss to spike. The remaining layers compensate, routing information around the dead layer via attention skip connections or residual paths. The compensation works — until the dead layer starts to matter, at which point performance degrades in ways that appear sudden but are not: the dead layer has been accumulating dead-layer failure for thousands of steps.

**The diagnostic signal**: per-layer gradient magnitude heatmap. Plot layer index (x-axis) against training step (y-axis) with color intensity proportional to average gradient magnitude in that layer during that training step. A healthy training run produces a roughly uniform color across the full grid. An unhealthy run shows dark (near-zero) bands indicating dead layers and bright (saturated) bands indicating exploding layers.

**The prescriptive fix**: Creus Castanyer et al. (2025) identify gradient-level interventions — specifically targeting the gradient distribution across layers rather than treating training instability as a learning-rate calibration problem. The insight is architectural: the fix is not to clip or scale the aggregate gradient but to explicitly regularize the distribution of gradient magnitudes across layers so that each layer maintains a minimum functional gradient throughout training.

(see Figure 3)

---

**Figure 3 — Gradient Flow Health Heatmap**

```
  ══════════════════════════════════════════════════════════════════════════════
   GRADIENT FLOW HEALTH ACROSS LAYERS AND TRAINING TIME
   (After Creus Castanyer et al., 2025)
  ──────────────────────────────────────────────────────────────────────────────

  UNHEALTHY TRAINING                  HEALTHY TRAINING
  ─────────────────────               ─────────────────────
  Training                            Training
  Step ▲                              Step ▲
       │ L1  L8  L16  L24  L32              │ L1  L8  L16  L24  L32
  2000 │ ░░░ ███ ░░░  ░░░  ███         2000 │ ▓▓▓ ▓▓▓ ▓▓▓  ▓▓▓  ▓▓▓
  1500 │ ░░░ ███ ░░░  ░░░  ███         1500 │ ▓▓▓ ▓▓▓ ▓▓▓  ▓▓▓  ▓▓▓
  1000 │ ░░░ ▓▓▓ ░░░  ░░░  ▓▓▓         1000 │ ▓▓▓ ▓▓▓ ▓▓▓  ▓▓▓  ▓▓▓
   500 │ ▓▓▓ ▓▓▓ ░░░  ░░░  ▓▓▓          500 │ ▓▓▓ ▓▓▓ ▓▓▓  ▓▓▓  ▓▓▓
     0 └──────────────────────►           0 └──────────────────────►
         Layer Index (shallow→deep)            Layer Index (shallow→deep)

  Legend:  ░░░ ≈ near-zero gradient (dead layer)
           ▓▓▓ ≈ moderate gradient (functional)
           ███ ≈ high-magnitude gradient (risk of overshooting)

  ANNOTATION: Dead layers (░░░) appear early and persist — they are not
  a training artifact but a permanent failure to learn. Layers immediately
  downstream of dead layers compensate, creating the false impression that
  training is healthy from the loss curve alone.
  ══════════════════════════════════════════════════════════════════════════════
```

*Figure 3: Gradient flow health heatmap. Left panel shows unhealthy training with persistent dead layers (near-zero gradient, white) and exploding layers (high gradient, black). Right panel shows healthy training with approximately uniform gradient magnitude across all layers. Neither pattern is visible in the aggregate loss curve. Creus Castanyer et al. (2025) identify gradient anisotropy as a primary driver of performance degradation when scaling deep RL networks.*

---

> **Key Takeaway:** Scaling deep RL networks produces gradient anisotropy — dead layers and exploding layers — that is invisible in the loss curve but detectable in a per-layer gradient magnitude heatmap. Creus Castanyer et al. (2025) establish that gradient distribution control, not learning rate tuning, is the correct lever for addressing this failure. Implement the heatmap monitor before scaling network depth.

---

## §4 — Isotropic Gaussian Representations

The gradient problem is in how signals flow through the network. The representation problem is in what the network has learned to store. Saheb, Obando-Ceron, Courville, Bashivan, and Castro (2026) identify a third failure mode: when RL training under non-stationary targets forces the agent to track moving objectives, the geometry of its internal representations determines whether it can adapt.

The core result is precise and worth stating in the paper's own terms: under non-stationary targets, isotropic Gaussian embeddings are provably advantageous. In particular, they induce stable tracking of time-varying targets for linear readouts, achieve maximal entropy under a fixed variance budget, and encourage a balanced use of all representational dimensions — all of which enable agents to be more adaptive and stable (Saheb et al., 2026).

Unpack each property:

**Stable tracking of time-varying targets for linear readouts.** RL agents continuously update their value estimates and policy targets as they learn. A representation that is anisotropic — elongated along some dimensions, compressed along others — will update smoothly along its principal dimensions but resist updates along its minor dimensions. This asymmetry means that when the learning objective shifts into a minor dimension (as it will, because RL targets are non-stationary and do not respect the geometry of the current representation), the agent resists the update. Isotropic representations, by definition, treat all directions equally — any update in any direction encounters the same representational capacity.

**Maximal entropy under a fixed variance budget.** A fixed variance budget is a proxy for a fixed representational capacity. The isotropic distribution extracts maximum information from that capacity by spreading it uniformly rather than concentrating it in a few directions. This is the representation-space analogue of the classic result that the Gaussian distribution maximizes entropy for fixed mean and variance: among all distributions with fixed variance per dimension, the isotropic Gaussian is the most expressive.

**Balanced use of all representational dimensions.** Empirically, deep RL networks develop a small-rank representation problem under non-stationary training: most of the representational variance concentrates in a small number of principal components, leaving the majority of dimensions effectively unused. This is precisely the dead-neuron problem from a linear-algebraic perspective. Isotropic regularization counteracts this collapse by continuously re-distributing representational capacity across all dimensions.

**The diagnostic signal**: the eigenvalue spectrum of the representation matrix. Compute embeddings on a held-out batch of states and examine the singular value distribution. A healthy representation has a relatively flat singular value spectrum — many dimensions contributing. A collapsing representation has a power-law spectrum — a few dominant dimensions and a long tail of near-zero dimensions.

**The prescriptive fix**: Saheb et al. (2026) propose a training objective augmentation that encourages isotropic Gaussian embeddings without requiring significant changes to the existing architecture or training pipeline.

(see Figure 4)

---

**Figure 4 — Representation Geometry: Anisotropic vs. Isotropic**

```
  ══════════════════════════════════════════════════════════════════════════════
   REPRESENTATION GEOMETRY IN DEEP RL
   (After Saheb et al., 2026)
  ──────────────────────────────────────────────────────────────────────────────

  ANISOTROPIC (collapsing)        ISOTROPIC (stable)
  ─────────────────────────       ─────────────────────────────────
  Singular                        Singular
  Value  ▲                        Value  ▲
    σ1   │ █                        σ1   │ ████
    σ2   │ ██                       σ2   │ ████
    σ3   │ █                        σ3   │ ████
    σ4   │ █                        σ4   │ ████
    σ5   │                          σ5   │ ████
    σ6   │                          σ6   │ ████
    σ7   │                          σ7   │ ████
    σ8   │                          σ8   │ ████
         └─────────────────►              └─────────────────────►
         Representation dims               Representation dims

  PROPERTY          ANISOTROPIC     ISOTROPIC
  ─────────────────────────────────────────────────────────────
  Target tracking   Slow (off-axis) Fast (all-direction)
  Entropy           Low (wasteful)  High (max for variance budget)
  Adaptability      Low             High
  Rank              Effective low   Effective high
  ──────────────────────────────────────────────────────────────
  Diagnostic: examine singular value spectrum of embedding matrix
  on held-out batch. Power-law → anisotropic collapse → apply fix.
  ══════════════════════════════════════════════════════════════════════════════
```

*Figure 4: Representation geometry comparison. Anisotropic representations (left) concentrate variance in a few dominant singular values, leaving most representational dimensions unused. Isotropic representations (right) distribute variance uniformly. Saheb et al. (2026) prove that isotropy is provably advantageous under the non-stationary targets that characterize RL training: faster target tracking, higher entropy under fixed capacity, and more stable adaptation as the learning objective evolves.*

---

> **Key Takeaway:** Anisotropic representations — where variance concentrates in a few dimensions — are a structural liability under RL's non-stationary targets. Saheb et al. (2026) establish that isotropic Gaussian embeddings are provably advantageous, providing stable tracking of moving objectives, maximal entropy, and balanced dimensional usage. Diagnostic: monitor singular value spectrum of state embeddings. Fix: apply the isotropic regularization they propose.

---

## §5 — Asymmetric PPO: Mini-Critics for LLM Reasoning

The fourth failure mode is architectural: the standard RL actor-critic framework, which works reliably at small scale, breaks at LLM scale because maintaining a critic network of comparable size to the policy is computationally prohibitive. Liu, Obando-Ceron, Lu, He, Wang, Su, Zheng, Castro, Courville, and Pan (2025) identify this gap and close it.

Standard PPO uses a critic network (value function) that provides a baseline for reducing variance in policy gradient estimates. At small scale — a few hundred million parameters — training a separate critic is tractable. At LLM scale, a critic of comparable size to the policy doubles memory and compute requirements. As Liu et al. (2025) note, conventional value functions are computationally expensive to train at LLM scale and often fail under sparse rewards and long reasoning horizons.

The pragmatic response — adopted by most RL4LLM systems — is to eliminate the critic entirely and replace it with simple average-advantage baselines (as in REINFORCE and related methods). This makes the training loop tractable but introduces a stability cost: without a critic, policy gradient variance is higher, and training under sparse rewards and long horizons is more erratic. The sensitivity to reward signal sparsity — which, as noted in §1, is a fundamental feature of reasoning tasks — is amplified without variance reduction from a functioning critic.

**The Asymmetric PPO (AsyPPO) approach**: Liu et al. (2025) observe that the computational burden of a critic is proportional to its size, not to the number of critics. A collection of small critics — each significantly smaller than the policy — can collectively provide richer variance reduction than a single average-advantage baseline while remaining computationally tractable. AsyPPO employs a set of lightweight mini-critics, each smaller than the full policy, whose combination restores the critic's variance-reduction role without the compute cost of a full-size critic.

The asymmetry is the key insight: the policy and critic do not need to be the same size. The critic needs to be informative, not large. A smaller critic can be informative if it is specialized — trained on a subset of the state space, focused on a specific reward component, or conditioned on different context features. Multiple specialized small critics provide coverage that a single averaged baseline cannot.

**The diagnostic signal**: policy gradient variance per training step. High variance — high standard deviation of the gradient estimate across mini-batches — indicates that the critic is failing to provide effective variance reduction. This is detectable by monitoring the standard deviation of policy update magnitudes. If this is consistently high with a critic baseline but would be expected to be lower, the critic is providing insufficient signal.

**The prescriptive fix**: replace the single large critic or average-advantage baseline with a set of lightweight mini-critics. AsyPPO (Liu et al., 2025) provides the specific framework for coordinating mini-critics in a way that remains compatible with standard PPO training loops.

(see Figure 5)

---

**Figure 5 — Asymmetric PPO: From Full Critic to Mini-Critics**

```
  ══════════════════════════════════════════════════════════════════════════════
   FROM FULL CRITIC TO ASYMMETRIC MINI-CRITICS
   (After Liu et al., 2025)
  ──────────────────────────────────────────────────────────────────────────────

  STANDARD PPO (at LLM scale: broken)
  ─────────────────────────────────────
  [Policy θ  ←────── Policy Gradient ──── Advantage A_t = r + γV(s') - V(s)]
  [Critic φ  ──────────────────────────── V(s) estimate]
                                          ↑
                                 [PROBLEM: |φ| ≈ |θ|, prohibitive]

  RLHF WITHOUT CRITIC (common practice: high variance)
  ─────────────────────────────────────────────────────
  [Policy θ  ←── Policy Gradient ──── A_t = r - mean(r) over batch]
                                       ↑
                      [PROBLEM: no variance reduction → erratic updates]

  ASYMMETRIC PPO (AsyPPO, Liu et al., 2025: efficient + stable)
  ──────────────────────────────────────────────────────────────
  [Policy θ   ← Policy Gradient ← Ensemble advantage estimate]
                                          ↑
  [Mini-Critic φ₁]  [Mini-Critic φ₂]  [Mini-Critic φ₃]
  [   |φ₁| << |θ|]  [   |φ₂| << |θ|]  [   |φ₃| << |θ|]
       Each critic is lightweight. Combined: richer signal than average baseline.
  ──────────────────────────────────────────────────────────────────────────────
  RESULT: Critic's variance-reduction role restored at a fraction of the
  compute cost. Compatible with standard PPO training loops.
  ══════════════════════════════════════════════════════════════════════════════
```

*Figure 5: The three critic regimes in LLM RL training. Standard PPO requires a full-size critic (computationally prohibitive at LLM scale). Eliminating the critic (common practice) restores tractability but amplifies policy gradient variance. AsyPPO (Liu et al., 2025) uses lightweight mini-critics whose combined estimate provides variance reduction comparable to a full critic at a fraction of the cost.*

---

> **Key Takeaway:** Eliminating the critic in LLM RL training — the standard pragmatic workaround — amplifies policy gradient variance, particularly under sparse rewards and long reasoning horizons. Liu et al. (2025) propose Asymmetric PPO, which replaces the single large critic with lightweight mini-critics that collectively restore variance reduction at tractable compute cost. Monitor policy gradient variance; if consistently high, switch to AsyPPO-style mini-critics.

---

## §6 — The Markovian Thinker: Linear Scaling of Reasoning Depth

The fifth failure mode is unique to RL for reasoning: as reasoning depth grows, the state space grows, and exponential state-space growth is an inherent source of RL instability. Aghajohari, Chitsaz, Kazemnejad, Chandar, Sordoni, Courville, and Reddy (2025) address this with an architecture-agnostic approach to scaling reasoning depth linearly rather than exponentially.

RL training on reasoning tasks exposes LLMs to progressively longer reasoning chains. As chains grow longer, each additional reasoning step extends the horizon, increasing the credit assignment problem and the sensitivity of the policy gradient estimate to early-trajectory decisions. This horizon sensitivity is manageable when reasoning chains are short; it becomes a primary source of training instability when reasoning chains are long.

The Markovian Thinker addresses this by imposing a Markovian constraint on the reasoning process: each reasoning step depends only on the current state, not the full history of prior steps. This constraint seems restrictive — human reasoning is not Markovian; it loops back, reconsiders, and integrates information across long contexts. But the constraint buys a specific, valuable stability property: the effective horizon of the RL problem is bounded. With a bounded horizon, credit assignment is tractable even for extended reasoning chains.

The "architecture-agnostic" qualifier is significant. The Markovian constraint is not a new architecture but a training-time constraint that can be applied to any model that supports step-by-step generation. It does not require modifying the attention mechanism or the token generation procedure. It requires modifying how the RL objective is formulated — specifically, how the value function is estimated and how the credit assignment is structured across reasoning steps.

**The diagnostic signal**: episode length distribution. As RL training proceeds on reasoning tasks, monitor the distribution of reasoning chain lengths. A healthy run produces a stable or slowly-growing length distribution. An unhealthy run produces explosive length growth — the policy learning to produce longer and longer chains because length correlates with receiving a reward, not because longer chains are more accurate. Explosive length growth is a sign that the horizon problem has taken hold.

**The prescriptive fix**: apply the Markovian Thinker constraint (Aghajohari et al., 2025) to bound the effective horizon. The paper establishes that linear scaling of reasoning depth is achievable under this constraint, trading full-history context for stable training dynamics.

(see Figure 6)

---

**Figure 6 — Reasoning Depth Scaling: Exponential vs. Linear**

```
  ══════════════════════════════════════════════════════════════════════════════
   REASONING DEPTH SCALING UNDER RL TRAINING
   (After Aghajohari et al., 2025)
  ──────────────────────────────────────────────────────────────────────────────

  Episode    ▲
  Length     │                                         ⤷ UNSTABLE SCALING
  (reasoning │                                   ✦ ✦  ✦  ✦
  chain      │                              ✦ ✦
  depth)     │                       ✦ ✦ ✦
             │                  ✦ ✦
         ●●  │ ●●●●●●●●●●●●●●●●  ⤶ MARKOVIAN THINKER: Linear O(n)
             │ (Bounded effective horizon)
             │
           0 └─────────────────────────────────────────────────────────────►
                          Training Steps →

  ─────────────────────────────────────────────────────────────────────────────
  UNSTABLE: Full-history reasoning under RL → horizon grows exponentially
            → credit assignment fails → training instability

  STABLE: Markovian constraint bounds effective horizon to O(n) per step
          → credit assignment tractable at any reasoning depth
  ══════════════════════════════════════════════════════════════════════════════
```

*Figure 6: Reasoning chain length under RL training as a stability signal. Unconstrained RL on reasoning tasks produces exponential episode-length growth (upper curve) as the policy learns length correlates with reward. The Markovian Thinker constraint (lower curve) bounds the effective horizon, achieving linear scaling of reasoning depth with training steps. Aghajohari et al. (2025) demonstrate that the constraint is architecture-agnostic — it applies to any model that supports step-by-step generation.*

---

> **Key Takeaway:** Exponential reasoning-chain growth under RL training is an instability signal, not a capability signal. Aghajohari et al. (2025) show that imposing a Markovian structure on the reasoning process bounds the effective horizon, achieving linear rather than exponential scaling of reasoning depth. Monitor episode length distribution; if it grows without bound, apply the Markovian constraint.

---

## §7 — On-Policy Parallelism and Data Collection

The sixth failure mode is often overlooked because it appears in the data collection infrastructure rather than in the model itself. Mayor, Obando-Ceron, Courville, and Castro (2025) examine how parallel data collection affects deep RL training stability and find a bias-variance trade-off that practitioners rarely measure.

The use of parallel actors for data collection has been effective across RL algorithms. Multiple environments running simultaneously increase data throughput and reduce wall-clock training time. But Mayor et al. (2025) find that the manner in which data is collected — controlled via the number of parallel environments and the rollout length — induces a form of bias-variance trade-off that interacts with training stability.

**The bias-variance trade-off in data collection**: more parallel environments produce more diverse data per batch, reducing variance in the policy gradient estimate. But they also introduce a lag between when the data was collected (with an older policy) and when it is used for training (with the current policy). This lag is off-policy bias: the training signal is computed under a policy that no longer matches the data-generating policy. At moderate parallelism, the bias is small and the variance reduction is worth it. At high parallelism, the bias can dominate, producing a training signal that systematically misleads the policy update.

**The number of training passes** introduces a related trade-off: more passes over collected data improve sample efficiency but risk overfitting on stale transitions — experiences that, under the evolved policy, would not be generated or would have different reward signals.

**The diagnostic signal**: monitor the discrepancy between the policy that generated the collected transitions and the current policy at the time of the update. Large discrepancy indicates high off-policy bias. Track this as a per-batch policy divergence metric (separate from the KL diagnostic in §2, which monitors the policy's distance from the reference SFT model; this monitors the policy's distance from the data-generating policy).

**The prescriptive fix**: tune the number of parallel environments and rollout length jointly with the number of training passes. Mayor et al. (2025) provide empirical analysis of these interactions. The practical finding is that simply adding more parallel environments without adjusting training passes produces worse stability, not better.

(see Figure 7)

---

**Figure 7 — Parallel Data Collection: The Bias-Variance Trade-off**

```
  ══════════════════════════════════════════════════════════════════════════════
   BIAS-VARIANCE TRADE-OFF IN PARALLEL DATA COLLECTION
   (After Mayor et al., 2025)
  ──────────────────────────────────────────────────────────────────────────────

  Training
  Stability
    ▲
    │           ★ Optimal Region
    │          ╱‾‾‾‾‾‾‾‾‾‾‾╲
    │         ╱               ╲
    │        ╱                 ╲
    │       ╱                   ╲
    │──────╱                     ╲──────────────────────────────
    │    Low                       High           Too High
    │   Parallelism               Parallelism      Parallelism
    │   [High variance:           [Optimal:         [High bias:
    │    few samples per          variance↓         off-policy
    │    batch, noisy             bias manageable]  lag dominates]
    │    gradient]

    └────────────────────────────────────────────────────────────►
                 Number of Parallel Environments →

  SECOND DIMENSION: Training Passes per Batch
  ──────────────────────────────────────────────────────────────────
  Too few passes  → sample inefficiency, slow learning
  Right number    → sample efficiency + generalization
  Too many passes → overfitting on stale transitions → instability

  Tune BOTH dimensions jointly. Mayor et al. (2025) identify the
  interaction as empirically significant and under-analyzed.
  ══════════════════════════════════════════════════════════════════════════════
```

*Figure 7: Stability as a function of parallelism level in on-policy data collection. Mayor et al. (2025) establish that both low and high parallelism degrade stability for different reasons — variance dominates at low parallelism; off-policy bias dominates at high parallelism. The optimal region must be found empirically for each setting. A second dimension — number of training passes over collected data — requires joint tuning to avoid overfitting on stale transitions.*

---

> **Key Takeaway:** Parallel data collection introduces a bias-variance trade-off that is independent of model architecture and often ignored. Mayor et al. (2025) demonstrate that the number of parallel environments and rollout length must be jointly tuned with the number of training passes. Simply scaling up parallelism degrades stability when off-policy bias begins to dominate. Monitor per-batch policy divergence from the data-generating policy as the primary diagnostic.

---

## §8 — The Hidden Instability: Policy Churn

The seventh failure mode is the least visible because it does not manifest in the states the policy visits during training. It manifests in the states it does not visit — and is nevertheless responsible for systematic instability in production RL systems.

Schaul, Barreto, Quan, and Ostrovski (2022) named this the **phenomenon of policy churn**: the greedy policy in DQN-style training changes rapidly across many states within just a few update steps. An update to the Q-function on a mini-batch of visited states changes the greedy action not only on those states, but on out-of-batch states that share representational features with the visited states. The greedy policy is churning — rewriting its action recommendations across the state space — with every mini-batch update, even though the gradient signal targets only a small subset of states.

The consequence is subtle but compounding: the policy the agent was trained to follow is not the policy the agent will follow at evaluation time, because evaluation covers out-of-batch states where the churn has been accumulating unrestricted. Schaul et al. (2022) observe a further property: churn functions as implicit exploration — the ε-greedy noise that practitioners assume is doing the exploration work may be less critical than the churn itself. The exploration behavior that RLHF and reasoning RL pipelines attribute to their exploration mechanism may be partially a churn artifact.

**The chain effect**: Tang and Berseth (2024) extend the Schaul et al. analysis to show that churn in the value function and churn in the policy interact: value churn induces policy churn, which induces further value churn. This **compounding chain effect** — the CHAIN paper's title naming it directly — means that churn self-amplifies. Left unchecked, a small amount of initial churn produces progressively larger divergence between the trained policy and the effective deployed policy. Tang and Berseth (2024) demonstrate that CHAIN reduces this compounding effect and is compatible with most existing RL algorithms, both online and offline.

**Bridging to LLM RL**: Li, Elmahdy, Boyd et al. (2025) translate the churn framework to multi-turn LLM agent training with SORL (Stabilizing Off-policy RL). The on-policy assumption in standard LLM RL post-training breaks down in multi-turn settings, where early conversation turns generate transitions under a policy that may be significantly different from the current policy at the time of training. SORL addresses this with turn-level importance sampling and clipping-triggered normalization that prevents training collapse in off-policy multi-turn settings. The approach is instantiated as SO-PPO and SO-GRPO, providing compatibility with the most common LLM RL training frameworks.

**Reward model stability under policy evolution**: Huang, Xia, Ren et al. (2026) identify a related failure mode on the reward side. As the policy drifts during RL training, the reward model's accuracy on the evolved policy distribution degrades — a form of reward model churn. R2M (Real-Time Aligned Reward Model) addresses this by dynamically leveraging the evolving hidden states of the policy network to keep the reward model aligned with the current policy distribution. This is complementary to the KL Comedy fix from §2: where §2 addresses the estimator's accuracy, R2M addresses the reward signal's accuracy as the policy evolves.

(see Figure 8)

---

**Figure 8 — The Policy Churn Chain Effect**

```
  ══════════════════════════════════════════════════════════════════════════════
   THE COMPOUNDING CHAIN EFFECT OF VALUE AND POLICY CHURN
   (After Schaul et al., 2022; Tang & Berseth, 2024; Li et al., 2025)
  ──────────────────────────────────────────────────────────────────────────────

  Mini-batch
  update on
  visited states
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ VALUE FUNCTION CHURN                                                    │
  │  Q(s, a) changes for out-of-batch states s' ∉ batch                    │
  │  (Schaul et al., 2022)                                                  │
  └──────────────────────────────┬──────────────────────────────────────────┘
                                 │  induces
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ POLICY CHURN                                                            │
  │  Greedy policy π(s') changes for same out-of-batch states               │
  │  → action recommendations rewritten at every update step                │
  └──────────────────────────────┬──────────────────────────────────────────┘
                                 │  induces further
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ COMPOUNDING DIVERGENCE (CHAIN effect, Tang & Berseth, 2024)            │
  │  Bootstrapped TD targets incorporate churn-affected Q values            │
  │  → error in value estimate grows at each bootstrap step                 │
  └──────────────────────────────┬──────────────────────────────────────────┘
                                 │  in LLM multi-turn settings
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ MULTI-TURN TRAINING COLLAPSE (SORL fix, Li et al., 2025)               │
  │  Early-turn policy no longer matches current policy                     │
  │  → off-policy bias accumulates → training signal breaks down            │
  │  Fix: turn-level importance sampling (SO-PPO, SO-GRPO)                  │
  └─────────────────────────────────────────────────────────────────────────┘

  DIAGNOSTIC: Track out-of-batch action divergence — fraction of
  states not in current mini-batch where greedy action changes
  per training step. Growing divergence = churn taking hold.
  ══════════════════════════════════════════════════════════════════════════════
```

*Figure 8: The compounding chain effect of value and policy churn (Tang & Berseth, 2024, building on Schaul et al., 2022). Each mini-batch update propagates changes to out-of-batch states through the shared representation, creating value churn that induces policy churn that feeds back into value estimation. Li et al. (2025) demonstrate the LLM extension: in multi-turn settings, this chain effect produces training collapse that turn-level importance sampling (SORL) prevents. Huang et al. (2026) extend the analysis to the reward model side with R2M.*

---

> **Key Takeaway:** Policy churn — the greedy policy changing rapidly on out-of-batch states — is a systematic RL instability that compounds through the value-policy feedback loop (Schaul et al., 2022; Tang & Berseth, 2024). In multi-turn LLM RL settings, it produces training collapse that standard on-policy assumptions miss (Li et al., 2025). The diagnostic is out-of-batch action divergence. The fixes are CHAIN for standard RL settings and SORL for multi-turn LLM settings.

---

## §9 — A Practitioner Checklist for Stable Deep RL

Seven failure modes. Seven diagnostic signals. Seven fixes. Applied in sequence, they form a complete stability protocol for deep RL at LLM scale. Here it is, copy-pasteable.

---

**The 5-Minute Pre-Training Stability Audit**

Before launching any large-scale RL training run, confirm that you have monitoring and mitigations in place for each of these:

```
□ 1. KL ESTIMATOR MONITORING (Shah et al., 2025)
     Instrument: Multiple KL estimators in parallel (not just one).
     Signal: Cross-estimator variance > threshold → treat as instability event.
     Fix: Use max(all estimators) as working KL. Cross-validate with:
          - Mean log-prob under reference model
          - Output token distribution entropy
          - Sentence embedding cosine similarity to reference outputs

□ 2. PER-LAYER GRADIENT HEATMAP (Creus Castanyer et al., 2025)
     Instrument: Per-layer gradient magnitude logger, log every N steps.
     Signal: Any layer with near-zero gradient magnitude persisting → dead layer.
             Any layer with gradient magnitude far above the cross-layer mean → explosion risk.
             [Threshold: calibrate to your architecture — no universal value is paper-specified.]
     Fix: Gradient distribution regularization (not aggregate clipping).
          Target: approximately uniform gradient magnitude across all layers.

□ 3. EMBEDDING ISOTROPY CHECK (Saheb et al., 2026)
     Instrument: Singular value decomposition of state/token embedding matrix
                 on held-out batch, logged every N steps.
     Signal: Power-law singular value spectrum (few dominant dimensions) →
             representation collapse.
     Fix: Apply isotropic Gaussian regularization to the representation layer.

□ 4. CRITIC VARIANCE MONITORING (Liu et al., 2025)
     Instrument: Standard deviation of policy gradient magnitude across mini-batches.
     Signal: High and rising gradient variance under sparse rewards/long horizons →
             critic is providing insufficient variance reduction.
     Fix: Switch from average-advantage baseline to AsyPPO mini-critics.

□ 5. EPISODE LENGTH DISTRIBUTION (Aghajohari et al., 2025)
     Instrument: Distribution of reasoning chain lengths during training.
     Signal: Monotonically growing episode lengths without accuracy improvement →
             horizon problem, not capability improvement.
     Fix: Apply Markovian Thinker constraint to bound effective reasoning horizon.

□ 6. OFF-POLICY BIAS TRACKING (Mayor et al., 2025)
     Instrument: KL between policy-at-data-collection and policy-at-update time,
                 per mini-batch (distinct from the reference-policy KL in check 1).
     Signal: Growing discrepancy between collection and update policies →
             off-policy bias accumulating.
     Fix: Tune number of parallel environments and training passes jointly.
          [Threshold for discrepancy KL not paper-specified — calibrate to your setup.]

□ 7. POLICY CHURN MONITORING (Schaul et al., 2022; Tang & Berseth, 2024)
     Instrument: For a held-out set of states not in the current mini-batch,
                 track what fraction change their greedy action per update step.
     Signal: A meaningfully growing fraction of held-out states changing their
             greedy action per step → churn taking hold.
             [Specific threshold not paper-specified — track the trend, not the absolute value.]
     Fix: Apply CHAIN regularization (standard RL) or
          SORL importance sampling (multi-turn LLM RL).
     For reward model drift: monitor reward model accuracy on evolved policy
     distribution. If accuracy declining, apply R2M (Huang et al., 2026).
```

---

**If you take nothing else from this series, take these 5 stability checks:**

> 1. **Monitor KL with multiple estimators, not one.** A single estimator fails in both directions and you will not know which failure you're in. (Shah et al., 2025)
> 2. **Watch per-layer gradients, not just the aggregate.** Dead layers are invisible in the loss curve and accumulate silently. (Creus Castanyer et al., 2025)
> 3. **Measure your representation's isotropy.** Anisotropic representations cannot track non-stationary RL targets — this is provable, not a heuristic. (Saheb et al., 2026)
> 4. **Don't drop your critic because it's expensive — use mini-critics.** Variance amplification under sparse rewards is worse than the compute cost of AsyPPO. (Liu et al., 2025)
> 5. **Measure policy churn on out-of-batch states.** This is the failure mode that gets you at evaluation time, not training time — it's invisible without the right instrument. (Schaul et al., 2022; Tang & Berseth, 2024)

---

> **Final Key Takeaways**
>
> 1. Standard RL stability intuitions — monitor the loss curve, clip gradients, tune the learning rate — are insufficient at LLM scale. The failure modes that matter at scale are invisible in aggregate metrics.
> 2. The Courville-group papers in this article form a diagnostic-to-prescriptive arc: first understand what is breaking (KL Comedy, Stable Gradients, Churn), then apply targeted fixes (Isotropic Representations, AsyPPO, CHAIN, SORL, R2M, Markovian Thinker).
> 3. The Shape of Thought connection [← A5]: what is true of reasoning traces is also true of the learning process itself — the shape of the gradient distribution, the geometry of representations, and the churn dynamics of the policy collectively determine whether RL training builds capability or erodes it.
> 4. The frontier systems connection [→ A9]: the stability choices that differentiate successful from failed large-scale RL runs — in DeepSeek-R1, ProRL, and others — map directly onto the toolkit in this article.

---

## § What Comes Next

This article closes Part III of the series: the engineering solutions to the problems diagnosed in A10 and A1 (benchmarks, plasticity loss) and theorized in A3 and A4 (world models, GVF scaffolding). The stability toolkit here is the practical prerequisite for everything in Part IV.

**[→ A9: Reasoning at Scale: Frontier Systems]** applies this toolkit to real systems. DeepSeek-R1, ProRL, MiniMax-M1, and related frontier reasoning systems each make specific stability choices — some applying the fixes described here, some learning the hard way why the fixes matter. A9 provides the empirical benchmark: what does prolonged RL training at scale actually buy, and where does it break down?

**[← A5: Shape of Thought]** is the theoretical complement: if you have not read A5, the connection between representation isotropy and reasoning distribution quality will be opaque. A5 establishes that reasoning capability lives in the distributional shape of reasoning traces; A7 establishes that the stability of RL training determines whether that distributional shape is preserved or collapsed.

The full series thesis, now complete in its engineering layer: fix the world model [A3, A4], stabilize the RL signal [A6, A7], and the plasticity crisis dissolves — not as a patch, but as an emergent property of a system that can update continuously without collapsing [A1, A10].

---

## References

[1] Shah, V., Obando-Ceron, J., Jain, V., Bartoldson, B., Kailkhura, B., Mittal, S., Berseth, G., Castro, P. S., Bengio, Y., Malkin, N., Jain, M., & Venkatraman, S. (2025). **A Comedy of Estimators: On KL Regularization in RL Training of LLMs.** *Mila – Québec AI Institute / Université de Montréal.* arXiv:2512.21852.

[2] Creus Castanyer, R., Obando-Ceron, J., Li, L., Bacon, P.-L., Berseth, G., Courville, A., & Castro, P. S. (2025). **Stable Gradients for Stable Learning at Scale in Deep Reinforcement Learning.** *Mila – Québec AI Institute / Université de Montréal / Google DeepMind.* arXiv:2506.15544.

[3] Saheb, A., Obando-Ceron, J., Courville, A., Bashivan, P., & Castro, P. S. (2026). **Stable Deep Reinforcement Learning via Isotropic Gaussian Representations.** *Mila – Québec AI Institute / Université de Montréal.* arXiv:2602.19373.

[4] Liu, J., Obando-Ceron, J., Lu, H., He, Y., Wang, W., Su, W., Zheng, B., Castro, P. S., Courville, A., & Pan, L. (2025). **Asymmetric Proximal Policy Optimization: Mini-Critics Boost LLM Reasoning.** *HKUST / Mila, Université de Montréal / Alibaba Group.* arXiv:2510.01656.

[5] Aghajohari, M., Chitsaz, K., Kazemnejad, A., Chandar, S., Sordoni, A., Courville, A., & Reddy, S. (2025). **The Markovian Thinker: Architecture-Agnostic Linear Scaling of Reasoning.** *Mila / Microsoft Research / McGill University / ServiceNow Research.* arXiv:2510.06557.

[6] Mayor, W., Obando-Ceron, J., Courville, A., & Castro, P. S. (2025). **The Impact of On-Policy Parallelized Data Collection on Deep Reinforcement Learning Networks.** *Mila – Québec AI Institute / Université de Montréal.* arXiv:2506.03404.

[7] Chandra, A., Agrawal, A., Hosseini, A., Fischmeister, S., Agarwal, R., Goyal, N., & Courville, A. (2025). **Shape of Thought: When Distribution Matters More Than Correctness in Reasoning Tasks.** *University of Waterloo / Mila, Université de Montréal / Microsoft Research India / Google DeepMind.* arXiv:2512.22255.

[8] Schaul, T., Barreto, A., Quan, J., & Ostrovski, G. (2022). **The Phenomenon of Policy Churn.** *DeepMind.*

[9] Tang, H., & Berseth, G. (2024). **Improving Deep Reinforcement Learning by Reducing the Chain Effect of Value and Policy Churn (CHAIN).**

[10] Li et al. (2025). **Stabilizing Off-Policy Training for Long-Horizon LLM Agents (SORL).**

[11] Huang et al. (2026). **Real-Time Aligned Reward Model beyond Semantics (R2M).**
