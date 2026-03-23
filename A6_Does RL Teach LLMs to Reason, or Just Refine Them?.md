# Does RL Teach LLMs to Reason, or Just Refine Them?

Article 6 of 12 | [RL][RE] | Anchor papers: Wu et al. arXiv:2507.14843 · Kim et al. arXiv:2502.01694 · Zhang et al. arXiv:2512.07783 · Yue et al. arXiv:2504.13837 | Series: Continual Intelligence

---

Every major AI lab is betting billions on RL post-training. Two papers published within months of each other in 2025 say opposite things about whether it works. One argues that reinforcement learning from verifiable rewards (RLVR) cannot escape the capabilities encoded in the base model — that no matter how much RL compute you add, you are amplifying what already exists. The other proves — mathematically, not just empirically — that chain-of-thought reasoning with search constitutes a provably richer computation class than the base model can access through greedy decoding. One of these positions is right. Here is how to tell which.

The short answer: they are both right, at different scales and under different conditions. Refinement dominates at small RL budgets; genuine computational enrichment emerges at prolonged RL scale, driven not by creation of new primitives but by the systematic composition and recombination of existing ones. RL is an amplifier before it is an inventor. The amplifier eventually builds things the original signal could not.

This matters enormously for AGI roadmaps. If RL is only refinement, then post-training capability is strictly bounded by pre-training quality, and the field should invest in pre-training over post-training. If RL creates new computation, then post-training is the lever — improving the reward signal, training duration, and architectural stability unlocks fundamentally new capabilities. The empirical and theoretical record assembled in this article supports a middle position: RL first reorganizes, then composes, and at sufficient scale, the compositions produce capabilities absent from the base model.

---

## §1 — The Premise: RL Seems to Improve Reasoning

The surface observation motivating this debate is genuine: models trained with RLVR score higher on reasoning benchmarks than their base model counterparts. DeepSeek-R1 and the o1 family are the canonical examples — reasoning chains that solve competition-level mathematics, pass legal bar exams, and generate correct multi-step code by allocating extended inference-time compute to search-driven reasoning. This is real capability improvement. The question is what kind.

Before entering the debate, the vocabulary needs to be precise. RLVR and RLHF are different mechanisms with different failure modes, and conflating them poisons the analysis.

- **RLHF (Reinforcement Learning from Human Feedback)**: the reward signal is a human preference model — a trained classifier that predicts which output a human rater would prefer. The signal is learnable but noisy, subjective, and gameable. A model optimizing RLHF learns to produce outputs that look like what humans rate highly, which is approximately but not exactly aligned with quality.

- **RLVR (Reinforcement Learning from Verifiable Rewards)**: the reward signal is a verifier — a ground-truth checker for a problem with a correct answer (mathematics, code, formal logic). The signal is binary (correct/incorrect) or continuous over a solution space, not subjective. A model optimizing RLVR learns to produce verifiably correct outputs in domains where ground truth exists.

Most of the evidence in this article concerns RLVR. The debate about whether RL "teaches reasoning" is primarily a debate about RLVR, because it is RLVR that has produced the visible reasoning improvements and RLVR that has received the most theoretical and empirical scrutiny.

The case for taking RLVR seriously is made most sharply by Han, Pari, Gershman, and Agrawal (2025), who argue that standard supervised LLMs — trained without any reward signal — exhibit a structural limitation in adaptive reasoning. Their experiments with algorithmic tasks in esoteric programming languages reveal that LLM reasoning trained purely on next-token prediction overfits to the surface patterns of training data and fails to generalize algorithmic understanding across novel contexts. The same models that appear to solve mathematical reasoning on standard benchmarks fail systematically when the representation of the problem changes — when numbers are written in a novel base, when operators are renamed, when the scaffolding of familiar problem formats is removed. The conclusion Han et al. (2025) draw is that reward-based training — training against verifiable outcomes rather than imitation of training sequences — is necessary for the adaptive generalization that the term "reasoning" implies.

This establishes the premise. RLVR is not merely a fine-tuning trick for improving benchmark scores; it is, on the available evidence, a qualitatively different training signal that drives qualitatively different generalization behavior. The question is not whether RLVR helps — it demonstrably does — but what kind of help it provides.

(see Figure 1)

---

**Figure 1 — The Refinement-vs-Computation Debate: Two Competing Models**

```
  ══════════════════════════════════════════════════════════════════════════
   MODEL 1: REFINEMENT                  MODEL 2: NEW COMPUTATION
   ("The Invisible Leash")              ("Metastable Dynamics")
  ──────────────────────────────────────────────────────────────────────────

   What RL does:                        What RL does:
   Amplifies existing capabilities      Induces genuinely novel reasoning
   already encoded in base model        paths absent from base model

   Mechanism:                           Mechanism:
   Reweights token probabilities        Enables phase transitions across
   toward correct solutions that        metastable CoT clusters — new
   base model could reach at high K     computation class, not just
   (pass@high-K → pass@1)              probability reweighting

   Ceiling:                             Ceiling:
   BASE MODEL CAPABILITY                None in theory; constrained by
   ─────────────────────────            composition depth in practice
   ↑ RL compute adds noise above
   this ceiling, not signal

   Evidence FOR:                        Evidence FOR:
   Wu et al. 2025 (Invisible Leash)     Kim et al. 2025 (Metastable Dynamics)
   Yue et al. 2025 (Does RL Incentivize) Yuan et al. 2025 (Skill Composition)
   Zhang et al. 2025 (Pre/Mid/RL)       Sun et al. 2025 (RL Grokking/DELTA)
                                        Liu et al. 2025 (ProRL)

   Evidence AGAINST:                    Evidence AGAINST:
   Kim et al. 2025 (new computation     Wu et al. 2025 (ceiling observed
   requires prolonged RL, not small     empirically at standard RL budget)
   RL budget — consistent with both)
  ──────────────────────────────────────────────────────────────────────────
   RESOLUTION: Both are right at different RL scales.
   Small budget: Refinement dominates. Prolonged budget: composition
   produces new computation from old primitives.
  ══════════════════════════════════════════════════════════════════════════
```

*Figure 1: The two competing models of RL-induced reasoning. The Refinement Model (left) treats RL as capability amplification bounded by base model knowledge; the New Computation Model (right) treats it as capability creation through novel reasoning paths. The resolution — that both operate at different RL scales — requires accepting that the Invisible Leash and Metastable Dynamics results are compatible, not contradictory.*

---

> **Key Takeaway:** RLVR and RLHF are different mechanisms with different failure modes — conflating them distorts the analysis. The premise that RL improves reasoning is empirically grounded; Han et al. (2025) show that purely SFT-trained models overfit to surface patterns and fail to generalize algorithmically. The question is what mechanism drives the improvement.

---

## §2 — Model 1: The Invisible Leash

The bluntest challenge to the "RL creates new reasoning" narrative comes from Wu, Xuan, Lu, Liu, Dong, Harchaoui, and Choi (2025), whose paper is titled — provocatively and deliberately — *The Invisible Leash? Why RLVR May or May Not Escape Its Origin.*

The invisible leash hypothesis: RLVR post-training cannot fundamentally extend a model's reasoning capabilities beyond what was already encoded in the base model during pre-training. The improvements observed after RLVR are real, but they are improvements in reliably surfacing correct solutions that the base model could already reach at high sampling temperatures (high pass@K), not the creation of solutions the model was incapable of producing before RLVR.

The mechanism underlying this hypothesis is straightforward. Base LLMs, sampled at temperature > 0, produce a distribution over outputs. Some of those outputs are correct solutions; many are not. The ratio of correct outputs to total outputs (pass@1) is typically low. RLVR training, which rewards correct outputs and penalizes incorrect ones, shifts this distribution: the probability mass on correct solutions increases, probability mass on incorrect solutions decreases, and pass@1 improves substantially. This is unambiguously a real improvement — the deployed model produces correct answers more reliably — but it is a redistribution of existing probability mass, not the creation of new capability. The leash is the base model's knowledge distribution. RLVR can tighten it, but (per this model) cannot extend it.

The practical implication is a ceiling: additional RLVR compute, beyond a certain point, produces noise rather than continued gains. The distribution has been optimized toward its correct-answer mass; further training begins to introduce mode collapse, reward hacking, or simply re-shuffles the already-optimized distribution without finding new solutions. The curve of reasoning performance versus RLVR compute rises steeply at first, then plateaus, then potentially degrades — the characteristic signature of a process hitting its upper bound.

(see Figure 2)

---

**Figure 2 — The Invisible Leash: Performance Ceiling Under RLVR**

```
  ══════════════════════════════════════════════════════════════════════════
   REASONING SCORE vs. RLVR TRAINING COMPUTE  (Wu et al., 2025)
  ──────────────────────────────────────────────────────────────────────────

  100% ┤                                      ━━━ BASE MODEL CAPABILITY
       │                          ╭───────────     (the leash)
   75% ┤               ╭──────────╯           ↑
       │          ╭────╯                      LEASH TENSION REGION:
   50% ┤     ╭────╯                           RL compute increases;
       │╭────╯                                reasoning score noise,
   25% ┤                                      not signal
       │
    0% ┤────┬────────┬────────────┬──────────────────────→
         Base    Small RL    Standard RL    Extended RL
                 budget      budget         budget
                                  ↑
              ─────────────────────────────────────────
              RLVR improves pass@1 here: probability
              mass shifts from wrong to right outputs
              ─────────────────────────────────────────

  Key distinction:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Base model pass@K=0 for problem class P  →  RLVR cannot rescue P  │
  │  Base model pass@K>0 for problem class P  →  RLVR can reliably     │
  │                                               surface P's solution   │
  └─────────────────────────────────────────────────────────────────────┘
  ══════════════════════════════════════════════════════════════════════════
```

*Figure 2: The Invisible Leash. RLVR post-training improves reasoning scores by redistributing probability mass from incorrect to correct outputs — but is bounded above by what the base model could generate at high sampling temperatures. When base model pass@K = 0 for a problem class, RLVR cannot produce solutions; it has nothing to amplify. At standard RL budgets the curve plateaus at the base model capability ceiling; extended RL compute enters a noise regime rather than producing continued gains.*

---

The direct experimental test of the leash hypothesis: take problem classes for which the base model has zero probability of generating a correct solution (pass@K = 0, for any K). Apply RLVR. If the leash is real, RLVR should fail to improve performance on these classes, because there is nothing to amplify. If RL creates new computation, RLVR should eventually discover solutions that the base model cannot produce at any sampling temperature.

This is the test that makes the invisible leash hypothesis falsifiable, and it is the test around which the subsequent evidence turns.

> **Key Takeaway:** The Invisible Leash hypothesis states that RLVR cannot extend base model capability — it can only redistribute probability mass toward existing correct solutions. The ceiling is the base model's pass@high-K. The critical test is whether RLVR can solve problem classes where the base model has pass@K = 0. Wu et al. (2025) provide the conceptual framework; the subsequent empirical papers provide the test results.

---

## §3 — Model 2: New Computation Through Metastable Dynamics

The formal counterargument to the Invisible Leash comes not from another empirical study but from a proof. Kim, Wu, Lee, and Suzuki (2025), in *Metastable Dynamics of Chain-of-Thought Reasoning: Provable Benefits of Search, RL and Distillation*, demonstrate that chain-of-thought generation with search constitutes a provably richer computation class than greedy next-token prediction — and that this enrichment has a specific mathematical structure.

The key conceptual innovation is the reframing of CoT generation as a **metastable Markov process**. Under this framework:

- **Easy reasoning steps** — routine algebraic manipulations, standard template applications — form **densely connected clusters** in the Markov chain. Within a cluster, the model moves fluidly: there are many high-probability transitions, and any given step can be reached quickly. These are the steps LLMs perform reliably without search.

- **Hard reasoning steps** — applying a non-obvious theorem, identifying the correct decomposition of a novel problem, making a key inference that unlocks the rest of the solution — create **sparse, low-probability edges between clusters**. The probability of making such a step via greedy decoding is small. The model is trapped in one cluster, unable to transition to the next without a low-probability jump.

The metastability creates phase transitions at longer timescales: given enough time (or enough parallel samples), even a low-probability edge is eventually traversed. But "enough time" under greedy decoding can be exponentially long. This is the formal sense in which complex reasoning problems are hard for LLMs: not that the model cannot produce the reasoning steps, but that greedy decoding is inefficiently exploring a space where the critical transitions are exponentially rare.

Under this framework, Kim et al. (2025) prove that **search over the CoT Markov process enables phase transitions that greedy decoding cannot achieve in polynomial time**. The search procedure — which can be implemented as best-of-N sampling, beam search, or RL-with-verifier — provides the mechanism for crossing sparse inter-cluster edges efficiently. This is new computation in a formal sense: the computation class accessible to search + CoT strictly contains the computation class accessible to greedy + CoT.

The result has a direct implication for the Invisible Leash debate: if CoT with search is in a strictly richer computation class than greedy decoding, then a model trained to prefer search-induced CoT paths — which is precisely what RLVR does when the reward is correct solution verification — has access to computations unavailable without RL. The leash, under this framework, is not the base model's knowledge distribution but the greedy decoding process. RL, by training the model to generate CoT that traverses inter-cluster transitions, enlarges the effective computation class.

The critical caveat — which preserves consistency with the Invisible Leash observation — is that accessing this richer computation class requires the model to have the component reasoning steps available as base-model primitives. Kim et al.'s proof does not claim RL creates new primitives from nothing. It proves that systematic search over the compositional space of existing primitives enables qualitatively new problem-solving. The base model must have the atomic reasoning capabilities; RL provides the combinatorial amplification.

> **Key Takeaway:** Kim et al. (2025) prove that CoT generation with search is a provably richer computation class than greedy decoding, by modeling CoT as a metastable Markov process where hard reasoning steps are sparse inter-cluster transitions. RL training that rewards verified solutions teaches models to navigate these transitions. This is formal new computation — not a contradiction of the Invisible Leash, but a specification of when and how RLVR escapes it.

---

## §4 — The Interplay of Pre, Mid, and RL Training

Before the theoretical debate can be resolved, a practical complication must be addressed: modern training pipelines are layered in ways that make it very difficult to know what RL is actually contributing.

Zhang, Neubig, and Yue (2025) at Carnegie Mellon University identify this as the central methodological problem in the RLVR reasoning literature. In *On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models*, they argue that three distinct training phases are being confounded, with the result that RL is systematically credited for improvements that may originate earlier in the pipeline:

**Pre-training** establishes the base knowledge distribution — what tokens the model has seen, what patterns it has absorbed, what problem formats it has been exposed to. For reasoning, pre-training determines the atomic capability inventory: the primitive operations that can be composed into longer chains of inference. This phase is opaque: large-scale pre-training corpora are not publicly disclosed, and the specific data mixture that produces a given base model's reasoning capabilities is not auditable.

**Mid-training** — domain specialization between pre-training and RL — is chronically underexamined. Mid-training typically involves high-quality domain-specific data (mathematical solutions, formal proofs, code) that substantially improves the model's reasoning before any RL signal is applied. The improvement from mid-training can be substantial, and because it precedes RL in the pipeline, it shows up as an apparent RL effect when pipeline ablations are conducted at only the pre-training/post-training boundary.

**RL training** receives the model after mid-training and applies the reward signal. The capabilities visible at the end of this phase reflect the cumulative effect of all three stages.

The central challenge Zhang et al. (2025) identify: because pre-training corpora are opaque, and mid-training is often not separately reported, the field lacks the controlled comparisons needed to isolate RL's contribution. A model trained on high-quality mathematical data during mid-training, then fine-tuned with RLVR, may show dramatic reasoning improvements that are attributable primarily to the mid-training stage — but the RL component gets the credit, because the mid-training baseline was never reported.

(see Figure 3)

---

**Figure 3 — Pre / Mid / RL Ablation: Where Reasoning Gains Actually Come From**

```
  ══════════════════════════════════════════════════════════════════════════
   REASONING PERFORMANCE ACROSS TRAINING STAGES (Zhang et al., 2025)
   [Schematic based on training pipeline analysis — values are illustrative
    of relative contribution pattern, not paper-reported percentages]
  ──────────────────────────────────────────────────────────────────────────

                    Math         Code        General
                    Tasks        Tasks       Reasoning
                  ─────────────────────────────────────
   Base model    ████████       ███████      ████████
   + Mid-train   ███████████████████████     ████████████████
                             ↑                        ↑
                  LARGE GAIN here            MODERATE GAIN here
                 (often attributed to RL)   (often attributed to RL)

   + RL          ████████████████████████   ████████████████████
                 ←──────→ ←──────────────→  ←──────→ ←─────────→
                  RL gain  Mid-train gain    RL gain  Mid-train gain
                  (smaller (larger than      (smaller  (larger)
                  than     commonly          than
                  assumed) reported)         assumed)

  ──────────────────────────────────────────────────────────────────────────
  KEY FINDING: The lack of controlled mid-training ablations means
  improvements attributed to RL in headline results may substantially
  reflect mid-training gains. RL adds real signal, but its independent
  contribution is smaller than commonly reported.
  ══════════════════════════════════════════════════════════════════════════
```

*Figure 3: Schematic of pre-training, mid-training, and RL contributions to reasoning performance across task categories, based on the pipeline analysis in Zhang et al. (2025). Mid-training — domain specialization on high-quality data, conducted before RL — provides substantial gains that are typically attributed to RL in headline results. The RL contribution is real but smaller and more targeted than commonly assumed when ablations are conducted without a mid-training baseline.*

---

The implications of Zhang et al.'s analysis cut both ways. For Invisible Leash proponents: the finding supports the argument that base model capabilities (established through pre-training and mid-training) are the primary driver, with RL providing marginal but real amplification. For New Computation proponents: the finding means the true RL contribution has been mismeasured, but it does not rule out genuinely new computation — it just means the experiments must be better controlled.

The practical consequence for the debate: any paper making strong claims about what RL does or does not add to reasoning must be evaluated against whether it controlled for mid-training. Papers that compare "base model + RL" versus "base model alone" without a "base model + mid-training" baseline are measuring a confounded effect.

> **Key Takeaway:** The pre/mid/RL training pipeline is systematically confounded in current literature. Zhang et al. (2025) identify mid-training as the underexamined stage: substantial reasoning improvements typically attributed to RL may originate in domain-specific mid-training data. Both Invisible Leash and New Computation conclusions drawn from unablated pipelines must be treated cautiously.

---

## §5 — Does RL Incentivize Reasoning? Empirical Tests

With the theoretical framework established and the methodological complications identified, what does the direct empirical evidence say?

Yue, Chen, Lu, Zhao, Wang, Song, and Huang (2025) at Tsinghua University put the central question in their title: *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* This is a direct empirical investigation of the Invisible Leash hypothesis: the question of whether RLVR can produce capabilities that the base model could not produce at high sampling temperatures.

The analysis reinforces the leash constraint at standard RL training budgets. In domains where the base model has no solution trajectory available (pass@K = 0 for all reasonable K), RLVR fails to produce solutions. This is the experimental signature of the Invisible Leash: RLVR is redistributing probability mass, and when there is no mass on correct solutions, there is nothing to redistribute. The finding is consistent with Wu et al. (2025)'s theoretical framing.

A closely related investigation comes from Yeo, Tong, Niu, Neubig, and Yue (2025), who systematically investigate the mechanics of long chain-of-thought (CoT) reasoning in *Demystifying Long Chain-of-Thought Reasoning in LLMs*. Their four main findings are critical for understanding what RL is actually doing when applied to reasoning:

1. **SFT is not strictly necessary but simplifies training.** Long CoT behavior can emerge from RL alone, but supervised fine-tuning provides a useful starting distribution that makes RL training more efficient. This is relevant to the attribution question: it means some "RL improvements" may partly reflect SFT-stage behavior.

2. **Reasoning capabilities emerge with training compute but are not guaranteed — reward shaping is crucial for stabilizing CoT length growth.** Without carefully designed reward functions, RLVR training produces unstable or regressive behavior rather than growing CoT capability. The reward engineering task is non-trivial.

3. **Scaling verifiable reward signals with filtered noisy web solutions shows strong out-of-distribution (OOD) generalization potential.** This is the most positive finding for the New Computation view: well-designed RLVR can generalize beyond training distribution, which pure refinement of existing capabilities would not predict.

4. **Core abilities like error correction are inherently present in base models but require significant compute and careful RL design to reliably incentivize.** This is a nuanced version of the Invisible Leash: the capability exists in the base model as a latent competency, but RL is doing real work to make it reliably accessible — not just redistributing obvious probability mass, but surfacing deeply latent behaviors.

These findings from Yeo et al. (2025) suggest a more granular version of the Invisible Leash: the leash is real, but it has slack — base models contain latent capabilities (error correction, self-verification, step-back reasoning) that are present in principle but require careful reward engineering and significant compute to reliably surface. RLVR's contribution is not just probability reweighting; it is capability activation.

The "Illusion of Thinking" debate, which touched the same empirical territory, requires a brief note. Lawsen (2025) analyzed Shojaee et al.'s claims about "accuracy collapse" in large reasoning models on planning puzzles and identified three critical methodological issues: Tower of Hanoi experiments likely exceeded model output token limits (the models were constrained by context length, not reasoning ability); the automated evaluation framework failed to distinguish reasoning failures from practical output constraints; and the River Crossing benchmarks included mathematically impossible instances for N ≥ 6, meaning models were scored as failures for not solving problems that have no solution. Lawsen's conclusion — that the findings primarily reflect experimental design limitations rather than fundamental reasoning failures — is a reminder that the empirical picture is more fragile than headline claims suggest, in either direction.

> **Key Takeaway:** Direct empirical tests confirm the Invisible Leash at standard RL budgets: RLVR cannot solve problem classes where base model pass@K = 0. However, Yeo et al. (2025) show that RL does more than redistribute probability mass — it activates latent capabilities (error correction, self-verification) that exist in the base model but require careful reward shaping and significant compute to reliably surface. The leash has slack.

---

## §6 — Prolonged RL: When Does It Cross the Threshold?

The Invisible Leash holds at standard RL budgets. But what happens at extended training regimes?

Liu, Diao, Lu, Hu, Dong, Choi, Kautz, and Dong (2025) at NVIDIA address this directly in *ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models*. The central claim is stated in the title: prolonged RL training — training regimes substantially longer than those used to produce standard reasoning models — does expand reasoning boundaries in ways that standard RL budgets do not.

The ProRL result has a specific character that distinguishes it from the standard Invisible Leash scenario. The capabilities that emerge under prolonged RL training are not merely improvements in pass@1 for problems the model could already solve at high K. They include expansion into problem classes and complexity regimes where the base model's initial solution probability was very low — approaching the pass@K = 0 boundary that defines the leash.

The mechanism that ProRL proposes for this expansion is consistent with Kim et al.'s metastable dynamics framework: prolonged training provides the compute budget needed to discover and reinforce the rare inter-cluster transitions that constitute hard reasoning steps. At standard RL budgets, the probability of spontaneously producing a correct inter-cluster transition is low, and the training signal for reinforcing it is correspondingly sparse. Prolonged training accumulates more of these rare events, giving the reward signal more data to work with, until the sparse transitions become robustly reinforced.

This is an important reconciliation point. The Invisible Leash is not a fixed ceiling independent of compute — it is a compute-dependent ceiling. At small RL budgets, the leash is tight. At prolonged RL budgets, the leash loosens as rare reasoning transitions are discovered, reinforced, and composed into reliable strategies. The ceiling rises with RL compute, but slowly, and only when the base model has the component capabilities that can be composed.

The practical consequence: labs with access to large RL training budgets can break through leash constraints that appear absolute to researchers with limited compute. This explains part of the divergence in the community's experience: practitioners at frontier labs observe that RL "creates new capabilities," while researchers using standard RL budgets observe a ceiling. Both observations are correct, at different scales.

> **Key Takeaway:** ProRL (Liu et al., 2025) demonstrates empirically that prolonged RL training expands reasoning boundaries in ways standard RL budgets do not. The ceiling imposed by the Invisible Leash is compute-dependent: rare inter-cluster reasoning transitions that are inaccessible at small budgets become reinforceable at large ones. The leash loosens with scale, but the component capabilities must exist in the base model.

---

## §7 — Grokking and Composition in RL

Two papers from September 2025 provide the mechanistic account of how prolonged RL eventually crosses the threshold that standard RL cannot.

Sun, Cao, Huang, Bai, Hajishirzi, Dziri, and Song (2025) at UC Berkeley introduce DELTA — Distributional Evaluation of Learnability and Transferrability in Algorithmic Coding — a controlled benchmark designed to probe two fundamental aspects of RL-induced learning: **learnability** (can RL solve problem families where pretrained models show pass@K = 0?) and **transferability** (if learned, does the capability generalize to structurally related but surface-different problems?). In *RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs?*, Sun et al. (2025) demonstrate that RL does exhibit a grokking-like phenomenon: after extended training, models that showed pass@K = 0 on specific algorithmic families abruptly develop the ability to solve them — a discontinuous phase transition rather than a smooth improvement curve.

The word "grokking" here is deliberate and specific. Grokking, in the deep learning literature, refers to the delayed generalization phenomenon where a model achieves near-perfect training accuracy but poor generalization, then — after additional training — abruptly generalizes. In the RL reasoning context, Sun et al. observe an analogous pattern: models train for extended periods without improvement on DELTA problem families with pass@K = 0, then cross a threshold and rapidly generalize. This is not redistribution of existing probability mass. At the moment of the grokking transition, the model is solving problems it genuinely could not solve before.

The DELTA results directly challenge the strongest version of the Invisible Leash hypothesis. If RL can unlock algorithms where base model pass@K = 0, the leash is not absolute. It is a transient barrier that prolonged training can eventually overcome — but only for problem families that have the right compositional structure.

The compositional account of what happens at the grokking transition is provided by Yuan, Chen, Zhang, Cui, Wang, You, Ding, Liu, Sun, and Peng (2025) in *From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones*. The central claim is captured in the title: RL teaches LLMs new skills not by inventing new primitives but by discovering and reinforcing compositions of existing primitives.

The f(g(x)) notation encodes the key insight. If a model knows function f (say, "identify a loop invariant") and function g (say, "simplify a recursive expression"), the model trained with SFT on examples of f and g separately does not automatically know f(g(x)) — the composed operation of simplifying a recursive expression and then identifying the invariant of the simplified form. This composition requires a new reasoning strategy, even though it uses no new primitives. RL training on problems that require f(g(x)) — and reward signal that distinguishes correct composed solutions from incorrect ones — can teach the composed strategy that pure imitation learning does not surface.

Yuan et al. (2025) frame the debate between Invisible Leash and New Computation as partially dissolved by this insight: RL does teach new skills (the compositions are genuinely new), but those skills are not ex nihilo — they are assembled from existing primitives. The leash is on the primitive inventory; the composition space above that inventory is unlocked by RL.

(see Figure 4)

---

**Figure 4 — Grokking and Composition: The Mechanism of RL's New Computation**

```
  ══════════════════════════════════════════════════════════════════════════
   DELTA BENCHMARK: RL GROKKING CURVE (Sun et al., 2025)
  ──────────────────────────────────────────────────────────────────────────

  Solve     ↑
  Rate      │                              ╭──────────────────────────────
  (pass@1)  │                         ╭───╯   ← grokking transition
            │                    ╭────╯
  ~0%       ┤────────────────────╯
            └───────────────────────────────────────────────────────→
               Base   Standard   Extended RL   Prolonged RL
               model  RL budget  budget        budget (ProRL regime)
            ↑
            pass@K = 0 at baseline: Invisible Leash in effect

   ─────────────────────────────────────────────────────────────────────
   SKILL COMPOSITION ACCOUNT (Yuan et al., 2025):

   BASE MODEL PRIMITIVES:            RL-LEARNED COMPOSITIONS:
   f(x): loop invariant ID           f(g(x)): simplify → invariant
   g(x): recursive simplification    h(f(x)): invariant → bound
   h(x): bound tightening            g(h(f(x))): compound strategy

   → New skills from composition of old ones
   → Each composition is genuinely new (not in base model's pass@K)
   → But bounded by available primitive inventory
  ══════════════════════════════════════════════════════════════════════════
```

*Figure 4: The grokking mechanism in RL-induced reasoning. Sun et al. (2025) show that models with pass@K = 0 at baseline exhibit a phase-transition-like grokking pattern under prolonged RL training — abruptly learning to solve problem families previously inaccessible. Yuan et al. (2025) provide the mechanistic account: RL learns compositions of existing primitives (f(g(x))), producing genuinely new problem-solving strategies that exceed pass@K = 0 for the base model on the composed task, while remaining bounded by the base model's primitive inventory.*

---

> **Key Takeaway:** Sun et al. (2025)'s DELTA benchmark demonstrates RL grokking: models exhibiting pass@K = 0 at baseline develop solution capability after extended training, via a discontinuous phase transition. Yuan et al. (2025) provide the mechanistic account — RL teaches new skills by discovering and reinforcing compositions of existing primitives (f(g(x))). This resolves the Invisible Leash / New Computation tension: new computation emerges at prolonged scale through composition, bounded by the primitive inventory established in pre-training.

---

## §8 — The Hivemind Risk: When RL Is Too Successful

A separate but related concern arises from NeurIPS 2025. Jiang, Chai, Li, Liu, Fok, Dziri, Tsvetkov, Sap, Albalak, and Choi (2025) at University of Washington and CMU publish *Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)*, a paper that asks not whether RL creates new reasoning but what happens when RL is applied uniformly across many models.

The Hivemind concern is distinct from the leash debate: it is not about whether individual models can develop new reasoning capability, but about what happens to the diversity of reasoning strategies across the ecosystem of models when all of them use the same RL recipes, reward functions, and training data. The argument is that RLVR, despite improving individual model capability, systematically reduces diversity across models — converging the ecosystem toward a homogeneous set of reasoning strategies that collectively represent a more fragile epistemic base than diverse, individually weaker models.

This is a systems-level failure mode rather than an individual-model failure mode. If the strongest version of the New Computation thesis is correct — that RL genuinely discovers new reasoning strategies — and all labs use the same RL recipe, then the "new strategies" discovered by RL converge across labs. The apparent diversity of frontier reasoning models conceals a high degree of strategic homogeneity: the same failure modes, the same blind spots, the same limitations, reproduced across nominally different systems.

The Hivemind paper (Jiang et al., NeurIPS 2025) is a structural warning about the AGI roadmap implications of the RL reasoning paradigm: even if RL succeeds at improving individual reasoning capability, uniform adoption of RL post-training across the ecosystem may produce a monoculture that is strategically brittle in ways that no individual model evaluation reveals.

> **Key Takeaway:** Jiang et al. (NeurIPS 2025) raise the Hivemind risk: uniform RLVR recipes applied across many labs converge model ecosystems toward strategic homogeneity. Even if RL successfully creates new reasoning strategies in individual models, the same RL procedures produce the same strategies across models — creating correlated failure modes at ecosystem scale that individual capability evaluations cannot detect.

---

## §9 — Verdict: A Unified Framework

The papers assembled above are not, at their core, contradictory. They map different regions of the same territory.

**The Invisible Leash is real at standard RL budgets.** Wu et al. (2025) and Yue et al. (2025) demonstrate empirically that RLVR's primary mechanism is probability mass redistribution toward correct outputs already present in the base model's distribution. For problem classes with base model pass@K = 0, standard RLVR fails to produce solutions. The ceiling is not fixed at the base model's pass@1 — it is fixed at the base model's pass@K for high K. RL narrows the gap between pass@1 and pass@high-K. This is real improvement; it is not new computation.

**The Metastable Dynamics result shows why RL can eventually escape the leash.** Kim et al. (2025) prove that CoT generation with search accesses a strictly richer computation class. The mechanism — phase transitions across metastable reasoning clusters — provides a formal account of how RL training at sufficient scale can produce capabilities that are unavailable to greedy decoding regardless of sampling temperature. The escape requires the component capabilities to exist as primitives; RL provides the combinatorial amplification over them.

**ProRL, DELTA, and the Skill Composition paper show when escape actually occurs.** Liu et al. (2025), Sun et al. (2025), and Yuan et al. (2025) collectively demonstrate that:
- The escape requires prolonged training (ProRL: not achievable at standard RL budgets)
- The escape manifests as grokking (DELTA: discontinuous phase transitions, not smooth improvement)
- The mechanism is composition (Yuan et al.: f(g(x)) — new strategies from old primitives)

**Zhang et al. (2025) provide the methodological calibration.** The training pipeline confounds pre/mid/RL contributions; claims about what RL does or does not add must be evaluated against mid-training baselines that are typically not reported.

**The Long CoT analysis (Yeo et al., 2025) provides the reward-engineering constraint.** Reward shaping is not a detail — it is a prerequisite. The same base model will exhibit dramatically different reasoning behavior under different reward functions. The quality of the RL outcome is strongly dependent on the quality of the reward signal.

**The Hivemind concern (Jiang et al., 2025) is the ecosystem-level caveat.** Even when RL succeeds at creating new computation, uniform RL recipes produce correlated reasoning strategies across models — a structural brittleness that individual capability metrics do not capture.

The unified framework that emerges:

*RL post-training is a compute-budget-dependent capability amplifier that operates primarily through probability mass redistribution at small budgets (Refinement Model) and through composition of existing reasoning primitives into novel strategies at prolonged budgets (New Computation Model). The transition between regimes is the grokking threshold — a phase transition rather than a smooth curve. The ceiling at any given compute budget is the base model's latent capability in the relevant primitive space, not an absolute bound. The quality of the RL signal (reward shaping) determines how efficiently this ceiling is approached and whether the grokking threshold is crossed.*

(see Figure 5)

---

**Figure 5 — Unified Framework: RL Reasoning Across Compute Regimes**

```
  ══════════════════════════════════════════════════════════════════════════
   RL REASONING: REGIME MAP ACROSS COMPUTE AND PRIMITIVE AVAILABILITY
  ──────────────────────────────────────────────────────────────────────────

  High  │     REGIME C: COMPOSITION
        │     Prolonged RL, sufficient primitives
  Base  │     ProRL, DELTA grokking, f(g(x))
  model │     New capabilities emerge through
  prim- │     composition; pass@K=0 → pass@1          ┌──────────────┐
  itive │     Mechanism: metastable phase transitions  │ NEW COMPUTE  │
  avail.│─────────────────────────────────────────────┤  (Kim 2025)  │
        │     REGIME B: AMPLIFICATION                  └──────────────┘
        │     Standard RL, sufficient primitives       ┌──────────────┐
        │     Visible benchmark improvements;          │  REFINEMENT  │
        │     base model pass@high-K → pass@1          │ (Wu et al.)  │
        │     Invisible Leash active                   └──────────────┘
        │─────────────────────────────────────────────
  Low   │     REGIME A: LEASH ACTIVE
        │     Any RL budget, insufficient primitives   ┌──────────────┐
        │     pass@K=0 throughout; RL noise only       │   NO GAIN    │
        │     Base model lacks primitive inventory     │  (leash hard)│
        └─────────────────────────────────────────────└──────────────┘
                  Small RL         Extended RL     Prolonged RL
                  budget           budget           budget (ProRL)
  ──────────────────────────────────────────────────────────────────────────
   Reward shaping quality determines transition efficiency across regimes.
   Mid-training (Zhang et al. 2025) expands the primitive inventory —
   moving problems from Regime A into Regime B/C.
  ══════════════════════════════════════════════════════════════════════════
```

*Figure 5: Unified framework for RL reasoning across compute regimes and primitive availability. At low primitive availability (base model pass@K = 0 for a problem class), RL produces noise regardless of compute budget — the hard Invisible Leash. At high primitive availability and small compute budget, RL amplifies existing capabilities (Refinement Model). At high primitive availability and prolonged compute budget, RL discovers compositional strategies (New Computation Model) through grokking transitions. Reward shaping quality and mid-training breadth are the two practical levers.*

---

**The Verdict**

RL does not teach LLMs to reason. It teaches them to reliably access reasoning they were already capable of — and, at sufficient scale, to compose those capabilities into strategies that are genuinely new. The distinction matters: *new* in the sense of "not available at any sampling temperature from the base model," but not new in the sense of "created from primitives the base model lacks."

This is a more qualified verdict than "RL creates reasoning" or "RL is just refinement." It is also a more useful verdict. It specifies exactly what RL can and cannot do:

- RL **can** improve pass@1 for problems where base model pass@high-K > 0
- RL **can** activate latent capabilities (error correction, self-verification) that exist in the base model but require careful reward engineering to surface
- RL **can** discover and reinforce compositional reasoning strategies at prolonged compute budgets
- RL **cannot** produce solutions to problems where the base model's primitive inventory is insufficient — this ceiling requires pre-training or mid-training to raise
- RL **cannot** replace careful reward engineering — reward shaping determines which regime the training enters

The AGI roadmap implication: if reasoning capability is the target, then the correct sequence is *expand primitive inventory through pre-training and mid-training, then apply prolonged RL with well-engineered reward signals*. Short-cutting to RL post-training without the foundational primitive base produces the amplification ceiling, not the compositional enrichment. And applying the same RL recipe across the ecosystem produces the Hivemind homogeneity risk even when individual models succeed.

> **Key Takeaway:** The verdict is that RL operates through two regimes: redistribution of existing capabilities at small compute budgets (Refinement Model), and compositional generation of novel strategies at prolonged compute budgets (New Computation Model). The transition between regimes is a grokking threshold determined by primitive availability, compute budget, and reward shaping quality. RL reorganizes and amplifies existing reasoning; it does not create reasoning from nothing, but composition creates strategies that are genuinely unavailable at baseline.

---

## § What Comes Next

This article argued that RL post-training operates in two regimes — refinement at small compute budgets, new computation through composition at prolonged budgets — and delivered a verdict: the Invisible Leash is real and compute-dependent, not absolute.

**[← A6: The Forgetting Transformer]** The analysis in this article depends on architectural stability as a prerequisite. A plasticity-impaired model trained with RLVR produces noise, not signal — the instability of the base architecture makes it impossible to distinguish genuine reasoning improvement from training artifact. FoX and the broader architectural programme in A6 are the necessary conditions for interpreting RL reasoning results cleanly. Without architectural stability, the RL compute budget is partly consumed by gradient interference and representational collapse rather than by reasoning improvement.

**[→ A5: Shape of Thought]** This article evaluated reasoning capability by accuracy and task performance. The next article reveals that accuracy is an insufficient metric: RL post-training simultaneously changes the *distribution* of reasoning traces — how models reason, not just whether they get the right answer. A model can increase its accuracy on a benchmark while its reasoning distribution collapses, becoming brittle and poorly generalizing. A5 asks the question this article does not: what does the reasoning look like, and is it getting better or worse on the dimensions that matter for generalization? [→ A5]

**[→ A7: Stable Deep RL at Scale]** The ProRL result — that prolonged RL training crosses the grokking threshold — requires stable training dynamics at extended scale. The toolkit for achieving stable deep RL (KL regularization, gradient norm management, policy churn control) is the subject of A7. Without stability, prolonged RL training produces divergence and mode collapse rather than grokking transitions. Stability is the practical prerequisite for observing the New Computation regime.

**[→ A9: Reasoning at Scale: Frontier Systems]** The Invisible Leash explains the ceiling that frontier systems like DeepSeek-R1 hit in extended inference scaling. DeepSeek-R1's impressive results are real — but they are bounded by the base model's primitive inventory, consistent with the unified framework developed here. A9 maps the frontier systems against this framework: which ones are in the Refinement regime, which have crossed the grokking threshold, and what the ceiling looks like at scale.

[← A6: Architectural stability is the prerequisite — unstable RL signal produces noise that looks like capability, not capability itself]

---

## Final Key Takeaways

1. **RLVR ≠ RLHF.** RLVR uses verifiable reward signals (ground-truth checkers); RLHF uses human preference models. The reasoning debate is primarily about RLVR. Conflating the two mechanisms produces category errors.

2. **The Invisible Leash is real at standard RL budgets.** RLVR's primary mechanism is probability mass redistribution toward correct outputs already present in the base model. For base model pass@K = 0, standard RLVR fails. The ceiling is the base model's latent capability.

3. **The Invisible Leash is not absolute.** Kim et al. (2025) prove that CoT with search is a provably richer computation class. ProRL (Liu et al., 2025) demonstrates empirically that the ceiling rises with prolonged training budgets.

4. **The grokking mechanism is the resolution.** Sun et al. (2025) show RL learning exhibits phase transitions: models that cannot solve a problem class at baseline abruptly develop the capability after prolonged training. Yuan et al. (2025) show the mechanism is skill composition — f(g(x)) from existing primitives f and g.

5. **The pipeline is confounded.** Zhang et al. (2025) demonstrate that mid-training contributions are systematically attributed to RL. Claims about RL's role in reasoning must be evaluated against mid-training baselines.

6. **Reward shaping is not a detail.** Yeo et al. (2025) show that reward shaping determines whether RL training stabilizes CoT length growth or collapses it. The quality of the RL signal determines which compute regime the training enters.

7. **The Hivemind risk is a systems-level concern.** Jiang et al. (NeurIPS 2025) demonstrate that uniform RL recipes converge model ecosystems toward strategic homogeneity. Individual capability gains do not preclude ecosystem-level brittleness.

8. **The practical sequence matters.** The correct order is: expand primitive inventory through pre-training and mid-training → apply prolonged RL with well-engineered reward signals. Shortcutting to RL without adequate foundational capability produces the refinement ceiling, not compositional enrichment.

---

## References

[1] Han, S., Pari, J., Gershman, S. J., & Agrawal, P. (2025). **General intelligence requires reward-based pretraining.** arXiv:2502.19402.

[2] Wu, F., Xuan, W., Lu, X., Liu, M., Dong, Y., Harchaoui, Z., & Choi, Y. (2025). **The Invisible Leash? Why RLVR may or may not escape its origin.** arXiv:2507.14843.

[3] Kim, J., Wu, D., Lee, J. D., & Suzuki, T. (2025). **Metastable dynamics of chain-of-thought reasoning: Provable benefits of search, RL and distillation.** arXiv:2502.01694.

[4] Zhang, C., Neubig, G., & Yue, X. (2025). **On the interplay of pre-training, mid-training, and RL on reasoning language models.** arXiv:2512.07783.

[5] Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., Song, S., & Huang, G. (2025). **Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model?** arXiv:2504.13837.

[6] Yeo, E., Tong, Y., Niu, M., Neubig, G., & Yue, X. (2025). **Demystifying long chain-of-thought reasoning in LLMs.** arXiv (2025).

[7] Liu, M., Diao, S., Lu, X., Hu, J., Dong, X., Choi, Y., Kautz, J., & Dong, Y. (2025). **ProRL: Prolonged reinforcement learning expands reasoning boundaries in large language models.** arXiv:2505.24864.

[8] Sun, Y., Cao, Y., Huang, P., Bai, H., Hajishirzi, H., Dziri, N., & Song, D. (2025). **RL grokking recipe: How does RL unlock and transfer new algorithms in LLMs?** arXiv:2509.21016.

[9] Yuan, L., Chen, W., Zhang, Y., Cui, G., Wang, H., You, Z., Ding, N., Liu, Z., Sun, M., & Peng, H. (2025). **From f(x) and g(x) to f(g(x)): LLMs learn new skills in RL by composing old ones.** arXiv:2509.25123.

[10] Lawsen, A. (2025). **The illusion of the illusion of thinking: A comment on Shojaee et al. (2025).** arXiv:2506.09250.

[11] Jiang, L., Chai, Y., Li, M., Liu, M., Fok, R., Dziri, N., Tsvetkov, Y., Sap, M., Albalak, A., & Choi, Y. (2025). **Artificial hivemind: The open-ended homogeneity of language models (and beyond).** *Advances in Neural Information Processing Systems (NeurIPS 2025)*. arXiv:2510.22954.
