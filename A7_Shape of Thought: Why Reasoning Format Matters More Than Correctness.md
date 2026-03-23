# Shape of Thought: Why Reasoning Format Matters More Than Correctness

Article 7 of 12  | [RE][RL] | Anchor papers: Kim et al. arXiv:2502.01694 · Zhang et al. arXiv:2512.07783 · Darlow et al. arXiv:2505.05522 | Series: Continual Intelligence 

---

A language model can be trained on reasoning traces where every single final answer is wrong and emerge stronger than one trained on traces where every answer is right. This is not a theoretical possibility. Chandra, Agrawal, Hosseini et al. (2025) demonstrate it empirically across three reasoning domains — mathematics, algorithmic reasoning, and code generation — on models ranging from 1.5B to 9B parameters across the Qwen, Llama, and Gemma families. The distribution of the reasoning traces, not the correctness of their final answers, predicts downstream performance. The shape of thought is not the answer. It is the path.

This result lands hard against a nearly universal assumption in LLM reasoning research: that data quality equals answer correctness. Dataset curation pipelines filter on correctness. Human annotation protocols reward correctness. RLVR post-training rewards nothing but correctness. The Shape of Thought finding suggests this entire orientation may be measuring the wrong thing.

The previous article in this series [← A2] established that RL's contribution to reasoning is more complicated than the headline results suggest: refinement dominates at standard training budgets, genuine capability expansion requires prolonged RL at the edge of competence, and the pre/mid/RL pipeline has been systematically confounded. This article takes the next step. Even if RL reliably improves answer accuracy, what does it do to the distribution of reasoning traces? Are accuracy and reasoning quality the same axis? The answer — supported by multiple lines of evidence — is no. They are independent axes. RL post-training typically moves along the accuracy axis. It frequently moves in the wrong direction on the trace-quality axis.

---

## §1 — What 'Shape of Thought' Means

Before the empirical evidence, the vocabulary needs to be precise. Chain-of-thought (CoT) reasoning has two distinct meanings in the LLM literature, and conflating them poisons the analysis.

**CoT as a decoding strategy**: a prompt engineering choice. The user inserts "let's think step by step" or similar scaffolding. The model generates intermediate reasoning tokens before producing its final answer. The model's parameters are unchanged; the strategy adds inference-time compute by conditioning the final answer on a generated reasoning prefix. Whether this helps depends on the base model's training distribution and the prompt structure. This is an inference-time choice, not a training-time change.

**CoT as a learned distribution**: what RLVR and supervised fine-tuning (SFT) on CoT traces actually optimize. The training process shapes which reasoning paths the model assigns high probability to. The model learns to prefer certain trace structures over others — particular lengths, formats, and compositional patterns — because those structures were reinforced during training. This is a property of the model's weight distribution, not a decoding choice.

These are different things with different failure modes. When researchers say "RL improves chain-of-thought reasoning," they almost always mean RL improves the accuracy of answers produced via the second definition — it shifts the learned distribution toward traces that produce correct outputs on the training distribution. What RL does to the trace distribution itself — its diversity, generalization capability, and OOD robustness — is a separate question that accuracy metrics do not answer.

Zhang, Neubig, and Yue (2025), in their controlled study of pre-training, mid-training, and RL interactions, provide a useful frame: RL's most reliable effect is improving **contextual generalization** — applying known patterns in novel surface contexts — rather than **extrapolative generalization** to fundamentally more complex problem structures. The distinction maps directly onto the two CoT definitions: CoT as learned distribution (contextual generalization from recognized trace formats) versus CoT as genuine novel computation (extrapolative generalization requiring new trace structures).

(see Figure 1)

---

**Figure 1 — The Reasoning Regime Spectrum**

```
  ══════════════════════════════════════════════════════════════════════════════
   THE REASONING REGIME SPECTRUM
  ──────────────────────────────────────────────────────────────────────────────

  ◄──────────────────────────────────────────────────────────────────────────►
    No Explicit      Short CoT          Long CoT           Latent Iteration
    Reasoning        (1–2 steps)        (multi-step)       (CTM — no tokens)

  ─────────────────────────────────────────────────────────────────────────────
  [Direct next-  [Prompt-level      [CoT as LEARNED DISTRIBUTION — what RL
   token          engineering        and SFT actually optimize]
   prediction]    choice — CoT       Hard steps require sparse inter-cluster
                  as DECODING        transitions (Kim et al., 2025)
                  STRATEGY]
  ─────────────────────────────────────────────────────────────────────────────

  What RL post-training typically does:

  ───────────────────────────────────────────────────────────────────────────►
  RL pushes distribution toward longer CoT regardless of whether longer
  traces generalize better — optimizing for accuracy, not trace diversity.

  ─────────────────────────────────────────────────────────────────────────────
  [ CTM → see A11 ]  At the extreme right: no token generation at all.
                      Per-neuron temporal memory replaces trace length.
                      The shape of thought becomes inaccessible.
  ══════════════════════════════════════════════════════════════════════════════
```

*Figure 1: The reasoning regime spectrum, from direct next-token prediction to token-free latent computation. The critical boundary is between CoT as a decoding strategy (a prompt engineering choice, left of center) and CoT as a learned distribution (a training-time property, right of center). RL post-training operates on the learned-distribution side and tends to push model behavior toward longer traces regardless of whether trace length predicts generalization. The CTM endpoint at the far right [→ A11] represents the logical limit where the trace disappears entirely.*

---

The shape of thought is the model's learned trace distribution: how it distributes probability mass across reasoning paths of different lengths, formats, and compositional structures. The Chandra et al. (2025) finding is striking precisely because it reveals that this distribution — not the answer it produces — is the more fundamental driver of reasoning capability.

> **Key Takeaway:** "Chain-of-thought" has two distinct meanings: a decoding strategy (inference-time prompt engineering) and a learned distribution (what RL and SFT optimize). Every empirical claim about what RL does to reasoning must specify which definition it concerns. The "shape of thought" refers to the learned distribution — its diversity, structure, and generalization properties. Answer accuracy is a downstream measure of this distribution, not a measure of the distribution itself.

---

## §2 — The Shape of Thought Paper

Chandra, Agrawal, Hosseini, Fischmeister, Agarwal, Goyal, and Courville (2025) set out to test a specific assumption: that answer correctness is the primary determinant of data quality for CoT training. Standard practice, as they note, relies on heavy human annotation or rule-based verifiers that filter model-generated traces by final-answer checking — the assumption being that correct answers index quality reasoning paths.

The core experiment: train a model on synthetic CoT traces generated by a more capable model, where all traces have been filtered to contain only *incorrect* final answers. Then compare this model against one trained on human-annotated traces with correct final answers, on held-out reasoning benchmarks.

The result: training on the incorrect-answer traces yields better performance on MATH, GSM8K, Countdown, and MBPP benchmarks across model families spanning 1.5B to 9B parameters. The conclusion Chandra et al. (2025) draw is precise: "a correct final answer is not always a reliable indicator of a faithful reasoning process."

Two mechanisms explain the finding.

**First: distribution proximity.** The synthetic CoT traces from a stronger model are generated by a language model, and therefore fall within the distributional space that language models naturally occupy. Human-annotated traces are generated by humans applying mathematical or logical reasoning, and carry stylistic and structural features that may be further from the student model's training distribution. A reasoning trace that matches the model's learned representation space may be easier to absorb — not because it is "better reasoning" in an objective sense, but because it requires less distribution shift. Chandra et al. (2025) test this directly by paraphrasing human-annotated traces — shifting them toward the model's distribution without changing their semantic content — and confirm that the distribution shift alone drives performance improvement.

**Second: partial validity.** The incorrect traces are not randomly wrong. They are generated by a more capable model that correctly executes most of the reasoning chain and only fails at the final step. Such traces typically contain sound intermediate structure — valid algebraic manipulations, correct problem decomposition, coherent logical steps — with only the final inference or computation wrong. The model learning from these traces absorbs the useful structural scaffold even though it absorbs an incorrect conclusion. The reasoning format is compositionally separable from the terminal answer: the scaffold can be valid even when the conclusion is not.

To test the tolerance boundary, Chandra et al. (2025) introduce increasingly flawed traces and measure how performance degrades. The decay is gradual rather than sudden — models absorb substantial structural information from traces before corrupted content begins to dominate. This tolerance curve gives practitioners a practical boundary: traces do not need to be correct, but they must retain the compositional architecture of valid reasoning.

(see Figure 2)

---

**Figure 2 — Accuracy vs. Reasoning Trace Diversity: Two Independent Axes**

```
  ══════════════════════════════════════════════════════════════════════════════
   ACCURACY vs. REASONING TRACE DIVERSITY
   (After Chandra et al., 2025 and Yue et al., 2025)
  ──────────────────────────────────────────────────────────────────────────────

   Trace
   Diversity  High ┤
   (breadth        │  ● Base LLM                  ★ Distillation
   of reasoning    │    (medium accuracy,            (higher accuracy +
   strategies)     │     rich distribution)          genuinely new patterns
                   │                                 introduced from teacher)
              Med  ┤
                   │
                   │                ● RL-trained
                   │                  (higher accuracy,
                   │                   compressed
                   │                   distribution)
              Low  ┤
                   │                              ● RL over-optimized
                   │                                (high accuracy,
                   │                                 collapsed diversity)
               0   ┤────────────────────────────────────────────────────────►
                      Low Accuracy              Medium            High Accuracy

  ─────────────────────────────────────────────────────────────────────────────
  CORE FINDING: Accuracy and trace diversity are independent axes.
  RL post-training moves reliably rightward (higher accuracy).
  It moves unreliably — and often leftward — on the vertical axis.
  The high-generalization region requires movement on BOTH axes.
  ══════════════════════════════════════════════════════════════════════════════
```

*Figure 2: Accuracy versus reasoning trace diversity as independent axes. Base LLMs occupy a medium-accuracy, high-diversity region — uncertain but compositionally rich. RL post-training pushes accuracy rightward reliably without preserving diversity. Over-optimized RL enters a collapsed-diversity zone where accuracy is high but OOD generalization is fragile. Distillation is the only method Yue et al. (2025) find that reliably introduces genuinely new trace patterns, moving toward the top-right region.*

---

The Shape of Thought paper's implication for dataset construction is direct. If distribution proximity predicts trainability better than answer correctness, then filtering synthetic datasets by answer correctness may be discarding valuable training signal while retaining traces that are correct but structurally misaligned with the student model's distribution. The practical guidance Chandra et al. (2025) offer: curate for structural coherence and distribution proximity, not terminal accuracy. A correct final answer is a noisy proxy for a useful training trace; structural fidelity to the model's learned representation space is the more reliable signal.

> **Key Takeaway:** Chandra et al. (2025) demonstrate empirically that the distribution of reasoning traces — not the correctness of their final answers — is the primary determinant of SFT data quality for reasoning tasks. Training on structurally coherent but answer-incorrect traces from a stronger model outperforms training on correct human-annotated traces across MATH, GSM8K, Countdown, and MBPP benchmarks. Two mechanisms: distribution proximity reduces learning difficulty; partial validity preserves the useful compositional scaffold.

---

## §3 — Theoretical Foundation: Metastable CoT Dynamics

A formal framework explaining why reasoning trace structure matters more than terminal correctness comes from Kim, Wu, Lee, and Suzuki (2025). Their paper, *Metastable Dynamics of Chain-of-Thought Reasoning: Provable Benefits of Search, RL and Distillation*, models CoT generation as a **metastable Markov process** and proves specific results about what search, RL, and distillation actually accomplish within this structure.

The framework distinguishes two types of reasoning steps:

- **Easy steps** — algebraic manipulations, standard template applications, routine format conversions — form **dense clusters** in the Markov chain. Within a cluster, transitions are numerous and high-probability. The model navigates these steps fluidly; they constitute the compositional core of the model's pre-trained trace distribution.

- **Hard steps** — applying a non-obvious theorem, identifying the correct problem decomposition, making a key bridging inference — create **sparse, low-probability edges between clusters**. Moving from one reasoning cluster to another requires crossing a transition that greedy decoding reaches only rarely.

This topology creates metastability at longer timescales: the system spends most of its generation budget within a cluster (performing easy reasoning) and only occasionally crosses to the next cluster (performing hard reasoning). The probability of a successful cross-cluster transition via standard autoregressive generation can be exponentially small — which is the formal reason that difficult multi-step problems are hard for LLMs even when the component reasoning steps are individually within the model's capability.

Kim et al. (2025) prove that **implementing a search protocol that rewards sparse inter-cluster edges improves CoT by decreasing the expected number of steps to reach different reasoning clusters.** This is not an empirical observation — it is a mathematical proof with a specific structural consequence: search genuinely expands accessible computation, for reasons traceable to the Markov topology of the reasoning process, not to the model's parametric knowledge.

(see Figure 3)

---

**Figure 3 — The Metastable Structure of Chain-of-Thought Reasoning**

```
  ══════════════════════════════════════════════════════════════════════════════
   METASTABLE CoT MARKOV PROCESS (Kim et al., 2025)
  ──────────────────────────────────────────────────────────────────────────────

  Cluster A                       Cluster B                   Cluster C
  (algebraic ops)                 (problem decomp.)           (theorem application)

  ┌──────────────┐                ┌────────────────┐          ┌────────────────┐
  │ ○─○─○─○─○─○ │                │ ○─○─○─○─○─○   │          │ ○─○─○─○─○─○   │
  │ ○─○─○─○─○─○ │                │ ○─○─○─○─○─○   │          │ ○─○─○─○─○─○   │
  │ ○─○─○─○─○─○ │ ~~~sparse~~~►  │ ○─○─○─○─○─○   │ ~sparse► │ ○─○─○─○─○─○   │
  │ ○─○─○─○─○─○ │ (rare, hard)   │ ○─○─○─○─○─○   │  (hard)  │ ○─○─○─○─○─○   │
  └──────────────┘                └────────────────┘          └────────────────┘
  Dense edges                     Dense edges                  Dense edges
  (frequent, easy)                (frequent, easy)             (frequent, easy)

  ─────────────────────────────────────────────────────────────────────────────
  GREEDY DECODING: spends most of its budget in Cluster A; rarely crosses
  sparse edges; hard reasoning steps reached with exponentially low probability.

  SEARCH (rewarding sparse edges): systematically explores inter-cluster
  transitions → decreases expected steps between clusters → reaches Cluster C.
  PROVED RESULT: search over CoT Markov process improves reasoning
  by decreasing expected cluster-transition time (Kim et al., 2025).

  RL (via policy gradient): fine-tunes the pretrained model toward sparse
  edges → the learned distribution prioritizes inter-cluster transitions.
  ══════════════════════════════════════════════════════════════════════════════
```

*Figure 3: The metastable structure of chain-of-thought reasoning, after Kim et al. (2025). Easy reasoning steps (algebraic manipulations, standard operations) form dense within-cluster transitions. Hard reasoning steps (applying a theorem, making a bridging inference) are sparse inter-cluster edges that greedy decoding reaches with exponentially low probability. Kim et al. (2025) prove that search provably decreases expected cluster-transition time; RL via policy gradient fine-tunes toward these sparse edges. The Markov topology — not terminal answer correctness — determines the utility of a reasoning trace.*

---

The metastable framework explains the Shape of Thought finding from §2. Why do structurally coherent but answer-incorrect traces train better than correct but distributionally distant traces? Because the metastable cluster structure determines which reasoning paths are compositionally available. A trace that exercises inter-cluster transitions — even if it commits an error in the final computation — provides information about where the sparse edges are. A correct but distributionally foreign trace may not: the model may have reached the right answer through a path that does not correspond to the target model's Markov topology, and that path is therefore harder to absorb.

Kim et al. (2025) also establish a formal limit complementing the Invisible Leash described in A2: **when the model is restricted to local information of the pretrained graph, its reasoning capability is bounded.** The base model's Markov topology sets the ceiling on what greedy decoding can achieve. RL escapes this limit not by inserting new nodes into the graph but by training the model to navigate existing inter-cluster edges that greedy decoding avoids.

> **Key Takeaway:** Kim et al. (2025) prove that CoT generation is a metastable Markov process where easy reasoning steps form dense clusters and hard steps are sparse inter-cluster edges. Search provably decreases expected transition time between clusters; RL fine-tunes the model toward sparse edges. The Shape of Thought finding makes sense in this framework: trace structure (which edges are exercised) predicts learning value better than terminal correctness (which node is reached).

---

## §4 — Gated Attention: Architecture Shapes Reasoning

The shape of a model's reasoning traces is not determined solely by training — architecture imposes its own constraints on which trace structures are representationally accessible.

The Qwen team (Qiu, Wang, Zheng, Huang et al., 2025) provide a systematic investigation in *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*, testing 30 variants of gating-augmented softmax attention across 15B Mixture-of-Experts and 1.7B dense model architectures trained on a 3.5 trillion token dataset. The central finding: applying a **head-specific sigmoid gate after the Scaled Dot-Product Attention (SDPA) operation** consistently improves model performance while simultaneously eliminating the **attention sink** phenomenon.

The attention sink is directly relevant to reasoning trace quality. In standard softmax attention, a small number of tokens absorb disproportionate attention weight — not because they are semantically important but because softmax forces a distribution that sums to one. These attention sinks act as low-information anchors that dilute the attention signal across the rest of the sequence. In multi-step reasoning, this dilution disrupts the model's ability to maintain coherent cross-step reference: earlier reasoning steps that should inform later ones may be under-attended when attention sinks are present.

Qiu et al. (2025) attribute the gate's effectiveness to two distinct mechanisms: first, the gate introduces a non-linearity on the low-rank mapping inherent in softmax attention, enriching the expressive capacity of each attention head; second, the query-dependent sparse gating scores selectively suppress uninformative attention mass, concentrating it on semantically relevant tokens. The result is attention that is both more expressive and more focused — a direct architectural intervention on the information-routing pathways that multi-step reasoning depends on.

(see Figure 4)

---

**Figure 4 — Attention Sinks vs. Gated Attention in Reasoning Contexts**

```
  ══════════════════════════════════════════════════════════════════════════════
   ATTENTION MECHANISM AND MULTI-STEP REASONING COHERENCE
   (After Qiu et al., 2025)
  ──────────────────────────────────────────────────────────────────────────────

  STANDARD SOFTMAX ATTENTION:            GATED ATTENTION (sigmoid after SDPA):
  (without gate)                         (head-specific sigmoid gate)

  Reasoning step 1  ─────────►           Reasoning step 1  ─────────►
  Reasoning step 2  ─────────►           Reasoning step 2  ─────────►
  Reasoning step 3  ─────────►           Reasoning step 3  ─────────►
  [Attention        ████████████         [Attention        ██░░░░░░░░
   sink token]      (absorbs most         sink token]       (gate suppresses
                     weight — softmax                        uninformative
                     constraint)                             mass)

  EFFECT: step 3 cannot attend           EFFECT: step 3 attends fully to
  coherently to steps 1 and 2.           prior reasoning steps.
  Multi-step derivations lose            Cross-step coherence preserved.
  referential integrity.

  ─────────────────────────────────────────────────────────────────────────────
  KEY FINDING: Head-specific sigmoid gate eliminates attention sinks,
  enhances long-context extrapolation, improves training stability,
  and tolerates larger learning rates.
  — Consistent improvement across 30 variants tested (Qiu et al., 2025)
  ══════════════════════════════════════════════════════════════════════════════
```

*Figure 4: Attention sinks versus gated attention in multi-step reasoning contexts, after Qiu et al. (2025). Standard softmax attention concentrates weight on uninformative sink tokens due to the unit-sum constraint, degrading cross-step coherence in long reasoning chains. A head-specific sigmoid gate after SDPA suppresses uninformative attention mass and restores full referential access across reasoning steps. The gate's effectiveness, confirmed across 30 architectural variants and 3.5T training tokens, comes from two mechanisms: non-linearity on softmax's inherent low-rank mapping, and query-dependent sparse gating.*

---

The connection to the series thesis is structural. The metastable cluster framework of Kim et al. (2025) describes which reasoning transitions are hard; gated attention determines whether the model's information routing can support those transitions. An attention mechanism that sinks weight into irrelevant tokens during a multi-step derivation is architecturally less capable of maintaining the inter-cluster coherence that hard reasoning steps require. Attention gates address the same representational fragility at the architectural level that search and RL training address at the optimization level.

> **Key Takeaway:** Qiu et al. (2025) demonstrate that a head-specific sigmoid gate after SDPA consistently improves performance and eliminates attention sinks across 30 variants tested at 15B MoE and 1.7B dense scale on a 3.5T token dataset. Attention sinks disrupt cross-step coherence in multi-step reasoning; sparse gating scores fix this by concentrating attention on semantically relevant tokens. Architecture shapes which reasoning trace structures are representationally accessible, independently of training.

---

## §5 — The Illusion of Thinking Debate

If reasoning trace quality is independent of accuracy, a critical measurement problem follows: how do you tell from the outside whether a model is genuinely reasoning or merely producing elaborate formatting that happens to correlate with correct answers on familiar benchmarks?

This question is at the center of the Illusion of Thinking debate. Shojaee et al. (2025) — whose paper is the target of Lawsen's (2025) reanalysis — report that Large Reasoning Models (LRMs) exhibit accuracy collapse on planning puzzles beyond certain complexity thresholds, interpreting this as evidence that elaborate thinking traces can masquerade as genuine reasoning while providing no systematic generalization. Models that appear to reason carefully at low complexity fail completely at higher complexity.

Lawsen (2025), in *The Illusion of the Illusion of Thinking: A Comment on Shojaee et al. (2025)*, challenges this interpretation by identifying three specific experimental design flaws.

**First: token limit conflation.** Tower of Hanoi experiments at high complexity risk exceeding model output token limits. The automated evaluation scores any incomplete solution as a failure. But Lawsen (2025) notes that model outputs at these instances often explicitly acknowledge the truncation — phrases like "the pattern continues, but to avoid making this too long, I'll stop here" — indicating that the model understands the solution structure but declines to enumerate it in full. This is a reasoning response. The automated framework scores it as a reasoning failure.

**Second: cannot-solve versus chose-not-to-enumerate.** The evaluation framework cannot distinguish between a model that cannot solve a problem and a model that can solve it but rationally chooses not to produce a 400-step move list. These are scored identically. The conflation systematically misclassifies capable models as failing on exactly the high-complexity instances where thinking models are most likely to invoke length management.

**Third: impossible benchmark instances.** Shojaee et al.'s River Crossing benchmarks include mathematically impossible instances for N ≥ 6 due to insufficient boat capacity. Models are scored as failing for not solving problems that are, by construction, unsolvable given the stated constraints. Lawsen (2025) confirms these instances are not borderline — they violate feasibility conditions provably.

When controlling for these artifacts — by asking models for generating functions rather than exhaustive move enumeration — Lawsen (2025) reports high accuracy on Tower of Hanoi instances that the original study classified as complete failures.

(see Figure 5)

---

**Figure 5 — The Evaluation Trap: Cannot Solve vs. Chose Not to Enumerate**

```
  ══════════════════════════════════════════════════════════════════════════════
   THE EVALUATION CHALLENGE IN HIGH-COMPLEXITY PLANNING TASKS
   (After Lawsen, 2025; responding to Shojaee et al., 2025)
  ──────────────────────────────────────────────────────────────────────────────

  SCENARIO A:                            SCENARIO B:
  Model CANNOT solve.                    Model CAN solve, but hits length limits.

  "I don't know how to..."               "The pattern continues, but to avoid
                                          making this too long, I'll stop here."

  Score under Shojaee et al.: FAIL       Score under Shojaee et al.: FAIL
  Correct score:              FAIL       Correct score:              PARTIAL PASS
                                          (model demonstrates understanding,
                                           declines exhaustive enumeration)

  ─────────────────────────────────────────────────────────────────────────────
  THREE CONFOUNDS (Lawsen, 2025):

  1. Output token limits         → Models truncate, not fail.
  2. Cannot-solve vs. chose not  → Automated frameworks cannot distinguish.
  3. Impossible instances        → River Crossing: N≥6 infeasible by construction.

  ─────────────────────────────────────────────────────────────────────────────
  FIX: Ask for generating function, not exhaustive move list.
  Result: High accuracy on instances previously scored as complete failures.
  ══════════════════════════════════════════════════════════════════════════════
```

*Figure 5: The evaluation trap in planning complexity benchmarks, after Lawsen (2025). Standard automated evaluation cannot distinguish between genuine reasoning failure and rational output management: models that understand Tower of Hanoi but decline to enumerate 400 moves are scored identically to models with no understanding. Three confounds identified by Lawsen (2025) — token limit conflation, cannot-solve/chose-not-to distinction, and mathematically impossible benchmark instances — explain the majority of the apparent accuracy collapse reported by Shojaee et al. (2025). Controlling for these confounds recovers high accuracy.*

---

The debate does not resolve the underlying question — whether elaborate thinking traces provide genuine generalization or elaborate formatting — but it establishes that the evidentiary bar for "LRMs don't really reason" is much higher than headline results suggest. The evaluation methodology must be able to distinguish capability failure from the rational decision not to enumerate a solution that exceeds output length constraints. Until that distinction is made, the "illusion of thinking" claim rests on a methodological artifact rather than a genuine capability finding.

> **Key Takeaway:** Lawsen (2025) demonstrates that Shojaee et al.'s "Illusion of Thinking" findings are substantially explained by three experimental design flaws: token limit conflation, evaluation frameworks that cannot distinguish cannot-solve from chose-not-to-enumerate, and mathematically impossible benchmark instances for N ≥ 6. The core measurement challenge — distinguishing genuine reasoning from pattern completion — remains unresolved, but requires evaluation methods that are robust to these confounds before drawing conclusions.

---

## §6 — Does RL Preserve the Shape?

The most direct empirical evidence that RL can improve accuracy while degrading reasoning trace quality comes from Yue, Chen, Lu, Zhao, Wang, Song, and Huang (2025) at Tsinghua University.

Their central finding: **RLVR-trained models outperform their base models at smaller values of k (e.g., k=1), but base models achieve higher pass@k scores when k is large.** This is the diagnostic signature of distribution compression. At k=1 — single sample — RLVR wins: it has concentrated probability mass on reliable solutions. At large k — many samples, high diversity — the base model wins: its broader distribution samples more of the solution space, including strategies that RLVR training compressed away.

The implication: RLVR narrows the reasoning distribution rather than expanding it. The base model's unoptimized distribution contains more solution strategies than the RLVR-optimized model. RLVR trades diversity for efficiency.

Yue et al. (2025) conduct a coverage and perplexity analysis that deepens this: **the reasoning paths generated by RLVR-trained models are already included in the base model's sampling distribution.** RLVR does not produce new reasoning strategies; it selects from the existing distribution for reliability. The reasoning capability boundary of LLMs often **narrows** as RLVR training progresses — the result is a model with higher pass@1 and lower pass@large-k than the base model it was trained from.

(see Figure 6)

---

**Figure 6 — RL Distorts the Reasoning Trace Distribution**

```
  ══════════════════════════════════════════════════════════════════════════════
   REASONING TRACE DISTRIBUTION: PRE-RL vs. POST-RL (Yue et al., 2025)
  ──────────────────────────────────────────────────────────────────────────────

  PRE-RL (Base Model):                   POST-RL (RLVR-trained):

  Frequency                              Frequency
     │                                      │
     │  ▁▂▄▅▆▇█▇▆▅▄▂▁                      │         ▂▄████▄▂
     │ /               \                    │        /          \
     │/                 \                   │       /            \
     ─────────────────────                  ─────────────────────
     Short ────── Long                      Short ────── Long
     Trace Length                           Trace Length

  Broad distribution across              Compressed around RL-favored
  reasoning strategies.                  trace format.
  HIGH pass@large-k.                     HIGH pass@1. LOWER pass@large-k.

  ─────────────────────────────────────────────────────────────────────────────
  KEY FINDING (Yue et al., 2025):
  RLVR reasoning paths are already contained within the base model's
  sampling distribution — RLVR selects from it, does not extend it.
  Six popular RLVR algorithms tested: consistent pattern, not an
  implementation artifact. Distillation alone introduces new patterns.
  ══════════════════════════════════════════════════════════════════════════════
```

*Figure 6: RLVR training compresses the reasoning trace distribution. Pre-RL base models maintain a broad distribution over reasoning strategies — high diversity, high pass@large-k. Post-RLVR, the distribution concentrates at RL-favored formats — high pass@1, but lower pass@large-k than the base model. Yue et al. (2025) establish through coverage and perplexity analysis that RLVR reasoning paths are already contained within the base model's sampling distribution, confirming that RLVR narrows rather than extends the reasoning distribution. The pattern holds across six popular RLVR algorithms.*

---

Yue et al. (2025) test six popular RLVR algorithms and find they perform similarly, all remaining far below the upper bound set by the base model's full potential. The uniformity indicates this is a property of the RLVR objective structure, not an artifact of any specific implementation. Training for one-hot correctness narrows the distribution; this is what one-hot correctness training does.

The distillation finding sharpens the contrast. Yue et al. (2025) demonstrate that distillation — copying reasoning traces from a teacher model — **can introduce genuinely new reasoning patterns from the teacher that expand the student's distribution**, unlike RLVR, which operates within the student's existing distribution. This is exactly consistent with the Shape of Thought result: a teacher's traces carry a different learned distribution, and exposing the student to that external distribution provides access to reasoning paths outside the student's pretrained space.

The combined picture: RL post-training is a distribution selector. It improves the reliability of sampling correct solutions from the existing distribution. It does not reliably expand the distribution. Accuracy improvement and distribution preservation are independent axes, and standard training practices conflate them.

> **Key Takeaway:** Yue et al. (2025) establish that RLVR post-training compresses the reasoning trace distribution — base models maintain higher pass@large-k than their RLVR-trained counterparts. The reasoning capability boundary narrows as RLVR training progresses, and RLVR-generated paths are already contained within the base model's sampling distribution. Six popular RLVR algorithms show this consistently. Distillation avoids the problem by injecting external trace diversity from a teacher.

---

## §7 — Beyond Format: When RL Shapes the Explanatory Scaffold

The analysis above establishes a problem: RLVR optimizes for correctness and compresses the reasoning distribution. What would it mean to optimize for the distribution itself — to train a model whose objective is the quality of the reasoning scaffold, not the correctness of the terminal answer?

Cetin, Zhao, and Tang (2025) at Sakana AI propose exactly this reframing. Their framework — **Reinforcement-Learned Teachers (RLTs)** — does not train a model to produce correct final answers. It trains a model to produce explanations that enable a *student model* to understand the solution.

Standard RLVR presents the model with a question and rewards correct answers. The exploration problem is severe: one-hot correctness provides no learning signal unless the model can already solve the problem at initialization, because there is no partial credit for almost-correct solutions. This creates a capability ceiling determined by initialization.

RLTs invert the objective. The teacher is presented with both the question *and* the solution, and is tasked to "connect the dots" — to produce a detailed explanation that makes the solution comprehensible to a student. The student's understanding is then tested, and the quality of that understanding provides dense reward feedback to the teacher. The teacher is not optimizing for its own correctness; it is optimizing for the quality of the reasoning scaffold it constructs for others.

(see Figure 7)

---

**Figure 7 — Standard RLVR vs. RLT Training Loop**

```
  ══════════════════════════════════════════════════════════════════════════════
   STANDARD RLVR (One-Hot Correctness)     RLT (Student Comprehension Reward)
   (Cetin, Zhao, Tang, 2025)
  ──────────────────────────────────────────────────────────────────────────────

  Question ──────────► Model              Question + Solution ──► Teacher (RLT)
                           │                                            │
                           ↓                                            ↓
                         Answer                                 Explanation
                           │                               (detailed scaffold)
                           ↓                                            │
                        Verifier                                        ↓
                      (correct?)                              Student model
                           │                             tests understanding
                           ↓                                            │
                    Reward: 1 or 0                                      ↓
                    (sparse signal)                         Dense reward signal
                                                            back to teacher

  ─────────────────────────────────────────────────────────────────────
  Failure mode: no learning signal          Advantage: teacher always
  when model cannot solve at               receives signal because
  initialization.                          teacher sees the solution.
                                           Signal measures explanation
  Ceiling: base model capability.          quality, not answer correctness.
  ══════════════════════════════════════════════════════════════════════════════
```

*Figure 7: Standard RLVR versus RLT training, after Cetin et al. (2025). RLVR rewards binary correctness — a sparse signal that provides no gradient when the model cannot solve the problem at initialization. RLTs are prompted with both question and solution and rewarded for producing explanations that enable student models to understand the problem. Dense comprehension-based rewards replace sparse correctness rewards, shifting the optimized quantity from answer accuracy to explanation quality.*

---

The empirical result is striking. Cetin et al. (2025) demonstrate that a 7B RLT provides higher final performance on competition and graduate-level reasoning tasks — measured on AIME, MATH, and GPQA benchmarks — than existing distillation and cold-starting pipelines collecting reasoning traces from models **orders of magnitude larger**. The mechanism is not that the 7B model discovers better solutions than a 70B model; it is that the 7B RLT has learned to construct reasoning scaffolds optimized for student transfer, which turns out to be more efficient than compressing the traces of a larger model.

Two properties of RLTs are particularly significant. First, they maintain their effectiveness when training students substantially larger than the teacher — including 32B models — suggesting that explanation quality, not teacher capability, drives downstream performance. Second, RLTs generalize zero-shot to out-of-distribution tasks. This is precisely the generalization property that RLVR-compressed distributions lack. A model trained to produce comprehensible explanations learns a reasoning format oriented toward transfer; a model trained to maximize one-hot correctness learns a format oriented toward shortcutting to the training distribution's answers.

This operationalizes the Shape of Thought thesis in its most direct form. If the distribution of reasoning traces is what matters — not the terminal correctness — then the right training objective is the quality of the scaffold that the trace provides to a learner, not the accuracy of the answer it reaches. RLTs are trained on exactly that objective, and the empirical result follows.

> **Key Takeaway:** Cetin et al. (2025) demonstrate that 7B RLTs — trained with dense comprehension-based rewards to produce explanations for student models — outperform distillation pipelines using orders-of-magnitude larger models on AIME, MATH, and GPQA. RLTs maintain effectiveness on 32B students and generalize zero-shot to OOD tasks. The key shift: optimizing for explanation quality rather than answer correctness produces reasoning formats oriented toward transfer, rather than formats that satisfy the training distribution.

---

## §8 — Beyond Format: When There Are No Tokens at All

The Shape of Thought argument assumes that reasoning traces exist as token sequences — things that can be measured, analyzed, and trained on. What happens at the logical limit of this argument, when the reasoning format disappears entirely?

Darlow, Regan, Risi, Seely, and Jones (2025) at Sakana AI introduce the Continuous Thought Machine (CTM), in which reasoning computation happens through **per-neuron temporal memory** rather than token generation. Each neuron maintains unique weight parameters that process incoming hidden-state histories over synchronization steps; the latent representation is a neural synchronization pattern rather than a token trace. There are no intermediate tokens to read, audit, or train a distribution over. [→ A11: Thinking Without Tokens]

The CTM sits at the far right of Figure 1's reasoning regime spectrum — the point at which the entire Shape of Thought framework applies differently. The shape of thought in a CTM is not a trace distribution; it is a synchronization pattern. Whether that makes the reasoning harder or easier to preserve, transfer, and evaluate is the question Article 11 examines in full.

> **Key Takeaway:** The CTM (Darlow et al., 2025) represents the logical limit of the reasoning-format question: when per-neuron temporal memory replaces token generation, the trace distribution studied in this article becomes inaccessible. What adaptive inference without tokens implies for generalization, interpretability, and training is the subject of [→ A11: Thinking Without Tokens: CTM].

---

## §9 — Implications for Evaluation and Training

The evidence assembled in this article has concrete consequences for how LLM reasoning should be measured and how post-training pipelines should be designed.

**Evaluation must go beyond pass@1.** The standard evaluation for reasoning LLMs — accuracy on a fixed benchmark at greedy or low-temperature decoding — measures pass@1, which is precisely the metric that RLVR training optimizes. Yue et al. (2025) show that pass@large-k is the more informative measure of reasoning capability breadth: it captures whether the model maintains a rich distribution of solution strategies or has compressed to a RLVR-favored subset. A model that improves on pass@1 but regresses on pass@large-k has improved reliability at the cost of reasoning coverage. For deployment contexts where novel problems are common, the coverage metric matters more.

**Distribution diversity should be a first-class metric.** The Shape of Thought finding and the Yue et al. (2025) distribution-narrowing result together suggest that the breadth of a model's CoT trace distribution — its learned distribution, not its decoding strategy — predicts OOD generalization better than pass@1. Metrics for trace diversity (variance in trace length, semantic coverage of reasoning strategies across samples, OOD format transfer rate) are underrepresented in standard evaluation protocols, even though they capture what accuracy metrics miss.

**Training signals should target the scaffold, not just the answer.** Zhang, Neubig, and Yue (2025) demonstrate that process-level reward signals — which reward correct intermediate reasoning steps, not just correct final answers — reduce reward hacking and improve reasoning fidelity. This is the training-side complement of the Shape of Thought finding: if the scaffold is what matters, reward the scaffold. Zhang et al. (2025) further find that RL produces genuine capability gains — up to +42% pass@128 when well-calibrated — only when pre-training leaves sufficient headroom and RL training data targets tasks at the model's edge of competence. RL applied to already-mastered problems compresses the distribution without extending capability.

**Contextual generalization requires minimal but sufficient pre-training exposure.** Zhang et al. (2025) find that contextual generalization — applying known reasoning patterns across novel surface contexts — requires minimal but sufficient pre-training exposure to the relevant task format; with sparse exposure of at least 1%, RL reliably transfers, yielding up to +60% pass@128 improvement. With near-zero exposure, RL fails regardless of RL compute. The implication: pre-training data coverage is a prerequisite, not an optional optimization.

(see Figure 8)

---

**Figure 8 — Shape-Aware Evaluation and Training Prescriptions**

```
  ══════════════════════════════════════════════════════════════════════════════
   SHAPE-AWARE EVALUATION AND TRAINING
  ──────────────────────────────────────────────────────────────────────────────

  EVALUATION:                              TRAINING:
  ────────────────────────────────────     ────────────────────────────────────
  Standard:  pass@1                        Standard: one-hot correctness (RLVR)
  → Measures reliability.                  → Compresses trace distribution.
  → What RLVR optimizes.                   → Narrows capability boundary.
  → Blind to distribution compression.    → Uniformly poor across 6 algorithms.

  Better:  pass@large-k                    Better: process-level rewards
  → Measures distribution breadth.         → Rewards intermediate steps.
  → Detects RLVR distribution collapse.    → Reduces reward hacking.
  → Predicts OOD generalization.           → Improves reasoning fidelity.
                                           (Zhang et al., 2025)
  Also:  trace diversity metrics
  → Length variance                        Best: RLT dense rewards
  → Semantic coverage of strategies        → Teacher sees question + solution.
  → OOD format transfer rate               → Dense comprehension signal.
                                           → 7B RLT > much-larger distillation.
                                           → Zero-shot OOD generalization.
                                           (Cetin et al., 2025)
  ══════════════════════════════════════════════════════════════════════════════
```

*Figure 8: Shape-aware evaluation and training prescriptions. Standard pass@1 evaluation and one-hot RLVR training form a consistent pair that optimizes reliability at the cost of distribution breadth. Process-level rewards and pass@large-k evaluation address the distribution dimension that standard protocols miss. The RLT framework is the most direct training operationalization of the Shape of Thought thesis: optimizing for explanation quality rather than answer correctness produces reasoning formats that generalize.*

---

> **Key Takeaway:** Evaluation must go beyond pass@1 to include pass@large-k and trace diversity metrics — the metrics that capture whether RL improved reasoning coverage or only reasoning reliability. Training should use process-level rewards (Zhang et al., 2025) and, where applicable, RLT-style comprehension-based rewards (Cetin et al., 2025) that directly target the reasoning scaffold rather than the terminal answer. RL applied at the model's edge of competence — not to already-mastered problems — is the only regime where genuine distribution expansion occurs.

---

> ## Final Key Takeaways
>
> 1. **Accuracy and reasoning trace quality are independent axes.** RL post-training reliably moves along the accuracy axis. Yue et al. (2025) demonstrate it often moves in the wrong direction on the trace-quality axis: pass@large-k decreases as RLVR training progresses.
>
> 2. **Distribution matters more than correctness.** Chandra et al. (2025) demonstrate that training on structurally coherent but answer-incorrect traces from a stronger model outperforms training on correct human-annotated traces across MATH, GSM8K, Countdown, and MBPP. The reasoning scaffold transfers; the terminal answer does not.
>
> 3. **The metastable structure explains why.** Kim et al. (2025) prove that hard reasoning steps are sparse inter-cluster Markov transitions that greedy decoding reaches with exponentially low probability. A trace exercising these transitions is valuable regardless of whether it terminates correctly.
>
> 4. **Architecture imposes its own constraints.** Qiu et al. (2025) demonstrate that attention sinks degrade multi-step coherence across 30 tested variants; gated attention that eliminates attention sinks is an architectural fix for the same representational fragility that RLVR training produces at the optimization level.
>
> 5. **Evaluation methodology determines what you can conclude.** Lawsen (2025) shows that apparent reasoning collapse in LRMs can be entirely explained by three experimental design flaws: token limit conflation, cannot-solve/chose-not-to confusion, and mathematically impossible benchmark instances. The underlying measurement challenge remains unsolved.
>
> 6. **RLVR narrows the reasoning distribution.** Six RLVR algorithms tested by Yue et al. (2025) consistently produce models that outperform base models at k=1 but underperform at large k. Distribution compression is a systematic RLVR property, not an implementation artifact.
>
> 7. **Optimizing for the scaffold changes what gets learned.** RLTs (Cetin et al., 2025) trained with dense comprehension-based rewards outperform distillation pipelines using much-larger models on AIME, MATH, and GPQA, and generalize zero-shot to OOD tasks. The objective function determines the trace distribution's purpose.

---

## § What Comes Next

This article has argued that reasoning format — the distribution of CoT traces in the learned-distribution sense — is more predictive of generalization than answer accuracy, and that RLVR post-training systematically degrades this distribution while improving accuracy. Two follow-up questions arise.

The first concerns training stability. Some of what appears as distribution compression may be training instability — gradient explosions, KL divergence spikes, policy churn — rather than a fundamental property of the RLVR objective. Distinguishing genuine distribution narrowing from instability artifacts requires the diagnostic and prescriptive toolkit assembled in [→ A7: Stable Deep RL at Scale]. If the reasoning distribution collapses during training, the first question is whether a stability fix recovers it before concluding that RLVR is inherently compressive.

The second is the limit case. The Shape of Thought argument assumes token traces as the substrate of reasoning computation. The CTM shows this is not necessary. When per-neuron temporal memory replaces token generation, the reasoning distribution studied in this article becomes a synchronization pattern rather than a trace-length distribution. What that means for evaluation, generalization, and the framework assembled here is the subject of [→ A11: Thinking Without Tokens: CTM].

---

## References

[1] Cetin, E., Zhao, T., & Tang, Y. (2025). **Reinforcement Learning Teachers of Test Time Scaling.** *Advances in Neural Information Processing Systems 38 (NeurIPS 2025)*. arXiv:2506.08388.

[2] Chandra, A., Agrawal, A., Hosseini, A., Fischmeister, S., Agarwal, R., Goyal, N., & Courville, A. (2025). **Shape of Thought: When Distribution Matters More Than Correctness in Reasoning Tasks.** *Preprint under review*. arXiv:2512.22255.

[3] Darlow, L., Regan, C., Risi, S., Seely, J., & Jones, L. (2025). **Continuous Thought Machines.** *Advances in Neural Information Processing Systems 38 (NeurIPS 2025)*. arXiv:2505.05522.

[4] Kim, J., Wu, D., Lee, J. D., & Suzuki, T. (2025). **Metastable Dynamics of Chain-of-Thought Reasoning: Provable Benefits of Search, RL and Distillation.** *Preprint*. arXiv:2502.01694.

[5] Lawsen, A. (2025). **The Illusion of the Illusion of Thinking: A Comment on Shojaee et al. (2025).** *Preprint*. arXiv:2506.09250.

[6] Qiu, Z., Wang, Z., Zheng, B., Huang, Z., Wen, K., Yang, S., Men, R., Yu, L., Huang, F., Huang, S., Liu, D., Zhou, J., & Lin, J. (2025). **Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free.** *Advances in Neural Information Processing Systems 38 (NeurIPS 2025)*. arXiv:2505.06708.

[7] Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., Song, S., & Huang, G. (2025). **Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?** *Preprint*. arXiv:2504.13837.

[8] Zhang, C., Neubig, G., & Yue, X. (2025). **On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models.** *Preprint*. arXiv:2512.07783.
