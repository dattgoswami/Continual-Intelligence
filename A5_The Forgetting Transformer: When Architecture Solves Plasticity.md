# The Forgetting Transformer: When Architecture Solves Plasticity

Article 5 of 12 | [CL][RL] | Anchor papers: Lin et al. arXiv:2503.02130 · Lin et al. arXiv:2504.06949 · Liu et al. arXiv:2505.24061 · Tang et al. arXiv:2506.00592 | Series: Continual Intelligence**

---

By task 2,000 in a sequential learning experiment, a deep ReLU network learns no better than a network with no hidden layers. That finding — reported by Dohare et al. in *Nature* (2024) — ended the era of training-procedure optimism in continual learning. Every gradient-based regularizer tested — L2, layer normalization, dropout, batch normalization — slowed the collapse but could not stop it. The only intervention that worked was non-gradient: random reinitialization of a tunable fraction ρ of the least-used neurons.

The next question was inevitable: if gradient descent cannot save plasticity, can architecture?

The answer is yes. But the mechanism is not what most researchers expected. The fix is not a new activation function or a larger weight matrix. It is a single gate inserted into the attention mechanism — one multiplicative term that converts the world's most widely deployed deep learning architecture into a natural continual learning substrate. This article explains how that gate works, why it solves the problem that training procedures could not, and how it sits at the centre of a broader architectural programme — adaptive computation pruning, gradient-based diagnostics, interference-reducing training dynamics, depth scaling, and inference-time weight editing — that collectively represents the first coherent engineering solution to the plasticity crisis.

---

## §1 — Two Approaches to Plasticity: Training vs. Architecture

The plasticity loss literature has, until recently, assumed that the fix must be a training-procedure fix. The assumption is understandable: neural networks are trained, after all. If they lose the ability to train, the natural place to look for the cure is in the training procedure.

The resulting interventions are effective up to a point. Continual backpropagation (Dohare et al., 2024) periodically reinitializes ρ (a tunable replacement rate) of the least-used neurons, injecting random diversity that gradient descent cannot provide. Spectral regularization keeps the singular value spectrum of weight matrices near its initialization distribution, preventing rank collapse. C-CHAIN (Tang et al., 2025, [→ §5]) reduces output variability for out-of-batch data, maintaining Neural Tangent Kernel (NTK) rank across the task sequence.

These interventions work. They are not wrong. But they share a structural limitation: they compensate for a flaw in the architecture they are applied to. Continual backpropagation reinitializes dead neurons; it does not prevent neurons from dying. Spectral regularization keeps weight matrices from rank-collapsing; it does not stop the collapse pressure from building. The mechanism driving plasticity loss — attentional saturation, gradient interference, representational collapse — continues operating underneath the compensatory intervention.

The architectural question is different: not "how do we compensate for the flaw?" but "how do we design an architecture that does not have the flaw in the first place?"

Standard Transformer attention has a specific, identifiable flaw in the context of continual learning. It was designed in 2017 for static input distributions — the full sequence is present, the distribution does not change, and there is no need to forget anything. Attend to everything that is relevant, and nothing is lost. This is the right design for a machine translation task on a fixed corpus. It is the wrong design for a system that must update its weights across thousands of sequential tasks, because the accumulated information never decays. Relevant contexts from task τ persist at the same weight as relevant contexts from task τ + 1,000. The network cannot distinguish what it needs to remember from what it should let go of.

This is the "why didn't we think of this sooner?" moment. The Transformer's lack of a forgetting mechanism was not an oversight — it was an intentional design choice for a problem where forgetting is harmful. Continual learning inverts the problem. For sequential task settings, the Transformer's perfect memory is its critical weakness.

[← A1: Continual backpropagation and spectral regularization are the training-procedure answers to the same problem the Forgetting Transformer solves architecturally]
[← A4: The Alberta GVF framework anticipated this structural separation — continual prediction targets provide a natural scaffold for what the agent should retain versus let decay]

> **Key Takeaway:** Training-procedure fixes compensate for plasticity loss; they do not prevent it. The root cause is architectural: standard attention was designed for static distributions and has no mechanism to decay old information. Architecture is the correct level of intervention — not because training tricks are wrong, but because they treat the symptom while the cause persists.

---

## §2 — The Forgetting Transformer

In early 2025, Lin, Nikishin, He, and Courville published a paper with a direct thesis: a forget gate can be naturally incorporated into softmax attention by down-weighting the unnormalized attention scores in a data-dependent way (Lin et al., 2025, ICLR).

The mechanism is precise. Recall standard causal softmax attention. The output **o**ᵢ at position i is a weighted sum over all preceding value vectors:

> **o**ᵢ = Σⱼ≤ᵢ exp(**q**ᵢᵀ **k**ⱼ) **v**ⱼ / Σⱼ≤ᵢ exp(**q**ᵢᵀ **k**ⱼ)

The numerator accumulates all previous key-value pairs weighted by query-key similarity. Nothing decays. Context from position 1 and context from position i are treated symmetrically, modulated only by their dot product.

The Forgetting Transformer (FoX) inserts a learned, data-dependent decay factor fᵢⱼ into the unnormalized score:

> **ã**ᵢⱼ = fᵢⱼ · exp(**q**ᵢᵀ **k**ⱼ)

where fᵢⱼ ∈ (0, 1] is computed from the input at position i and decays as a function of distance (i − j). This is the forget gate. It is not a fixed schedule — it is input-dependent, meaning the model learns to forget more or less depending on the current context. Causally relevant information survives; information that is not relevant to current processing, decays.

The practical results are significant across multiple evaluation axes (Lin et al., arXiv:2503.02130). FoX outperforms the standard Transformer on long-context language modeling, on length extrapolation, and on short-context downstream tasks, while performing on par with the Transformer on long-context retrieval tasks. On the needle-in-the-haystack test — the canonical measure of long-context retrieval — FoX achieves near-perfect accuracy within its training context length. Recurrent sequence models (Mamba-2, HGRN2, DeltaNet), which also have gating mechanisms, fail this test due to their smaller fixed-size hidden states. Three additional deployment properties matter: FoX requires no positional embeddings (the forget gate's distance-dependent decay provides implicit positional information), it is compatible with FlashAttention without algorithmic modification, and the accompanying "Pro" block design — incorporating architectural components common in recurrent models — substantially improves performance for both FoX and the baseline Transformer.

The connection to continual learning is direct. The forget gate provides a natural mechanism for the network to reduce the influence of activations from distant context — including activations shaped by prior tasks — while retaining the capacity to process local context fully. The network does not lose memory; it learns to weight memories differentially. This is precisely what continual backpropagation was attempting with random reinitialization, but achieved through architecture rather than a training-time intervention. The architectural fix is inherent to every forward pass; the training-procedure fix is applied on top of a broken architecture.

(see Figure 1)

---

**Figure 1 — Standard Softmax Attention vs. Forgetting Transformer (FoX)**

```
  ══════════════════════════════════════════════════════════════════════════
    STANDARD SOFTMAX ATTENTION            FORGETTING TRANSFORMER (FoX)
  ──────────────────────────────────────────────────────────────────────────

  Inputs: q_i, k_j, v_j                Inputs: q_i, k_j, v_j, plus learned
  for all j ≤ i                        forget gate f_ij for all j ≤ i

    Unnormalized score:                   Unnormalized score (modified):
    ã_ij = exp(q_i^T k_j)                ã_ij = f_ij · exp(q_i^T k_j)
                                                  ↑
                                          DATA-DEPENDENT FORGET GATE
                                          f_ij ∈ (0,1]: decays with distance
                                          f_ii = 1.0 (current token: no decay)
                                          f_i,i-k → 0 as distance k grows

    Output:                               Output:
    o_i = Σ_j ã_ij v_j                   o_i = Σ_j ã_ij v_j
          ─────────────                         ─────────────
          Σ_j ã_ij                              Σ_j ã_ij

  ──────────────────────────────────────────────────────────────────────────
  RESULT: All past context treated       RESULT: Past context decays in a
  symmetrically. No decay mechanism.     data-dependent way. Less-relevant
  Representational capacity fills        or distant activations down-weighted
  monotonically across tasks.            automatically at each position.
  ──────────────────────────────────────────────────────────────────────────
  NO POSITIONAL EMBEDDINGS REQUIRED  ·  FLASHATTENTION COMPATIBLE
  ══════════════════════════════════════════════════════════════════════════
```

*Figure 1: Standard softmax attention versus the Forgetting Transformer (FoX). The forget gate fᵢⱼ is the minimal structural addition: a learned, data-dependent decay applied to unnormalized attention scores before the softmax. This single multiplicative term converts a static-task architecture into one with a natural mechanism for differential information retention — the core structural requirement for plasticity preservation in continual learning.*

---

> **Key Takeaway:** The Forgetting Transformer adds one learned decay factor to softmax attention. This minimal change gives the network an explicit mechanism to downweight old, less-relevant information — precisely the mechanism standard attention lacks in continual settings. FoX outperforms the Transformer on long-context modeling, requires no positional embeddings, and is FlashAttention-compatible.

---

## §3 — Adaptive Computation Pruning: Making the Forget Gate Efficient

A natural efficiency objection: if many attention heads in FoX learn to forget quickly — concentrating entirely on local context — they are expending full quadratic attention computation on key-value pairs whose effective weights collapse to near-zero. Why compute attention over positions whose contribution is negligible?

Lin, Obando-Ceron, He, and Courville addressed this directly in a follow-up published at COLM 2025. The observation motivating Adaptive Computation Pruning (ACP): many attention heads in FoX develop strong local structure — their forget gate values decay rapidly enough that the effective attention window is far shorter than the full context length. These "local heads" contribute only local context. Their key-value computations for distant tokens are mathematically negligible; they satisfy a provable bound. "Global heads" — whose forget values decay slowly — continue to require full context and are not pruned.

ACP exploits this structure with a provably safe threshold mechanism. A dynamically computed pruning threshold τ_prune guarantees that any pruned attention weight satisfies |ã_ij| < ε for a negligible constant ε, regardless of the specific input values at those positions. The threshold is set dynamically per head based on each head's learned forget gate behavior, not statically. Once the forget gate decay crosses this threshold at distance (i − j) for query i, the computation is pruned with a formal safety guarantee.

The resulting computational structure is a natural sliding-window pattern: local heads process only a short recent context; global heads continue processing the full sequence. ACP makes this structure explicitly computable (Lin et al., arXiv:2504.06949). Applied to FoX pretraining across model sizes from 125M to 760M parameters and context lengths from 4k to 16k tokens, ACP consistently prunes around 70% of FLOPs and memory accesses in softmax attention. This produces a roughly 50% to 70% reduction in attention runtime — a 2–3× speedup — and a roughly 10% to 40% increase in end-to-end training throughput. Longer context lengths yield greater computational savings, because local heads have more distant tokens to prune. All speed improvements are achieved without any degradation in language modeling loss or downstream task performance.

The architectural implication is significant: the forget gate does not merely improve task performance — it creates a computable sparsity structure that translates directly into efficiency gains. The architectural fix and the efficiency fix are the same fix.

(see Figure 2)

---

**Figure 2 — Adaptive Computation Pruning: Local vs. Global Heads**

```
  ══════════════════════════════════════════════════════════════════════════
   CONTEXT WINDOW: position 1 ──────────────────────────────→ position N
  ──────────────────────────────────────────────────────────────────────────

   LOCAL HEAD (fast forget gate decay):
   Forget gate f_i,j:
   1.0 ████████████▓▓▓▒▒░░ → ~0 (threshold crossed here)
        │← effective window→│←── pruned by ACP (ε-negligible) ─────────→│
         (compute here only)

   GLOBAL HEAD (slow forget gate decay):
   Forget gate f_i,j:
   1.0 ████████████████████████████████████████████████████████ → 0.5+
        │← full context required: no pruning possible ──────────────────→│

  ──────────────────────────────────────────────────────────────────────────
   ACP RESULTS (Lin et al., COLM 2025):
   Model sizes:         125M → 760M parameters
   Context lengths:     4k → 16k tokens
   FLOP reduction:      ~70% of attention FLOPs and memory accesses
   Attention speedup:   ~2–3× (50–70% runtime reduction)
   Throughput gain:     ~10–40% end-to-end training throughput
   Longer context:      greater savings (more distant tokens to prune)
   Performance:         zero degradation in language modeling or downstream
  ══════════════════════════════════════════════════════════════════════════
```

*Figure 2: Adaptive Computation Pruning (ACP) classifies FoX attention heads as "local" (fast forget gate decay) or "global" (slow decay). Local heads have a provably negligible effective window beyond a threshold distance; ACP skips those computations with a formal correctness guarantee. Across model sizes of 125M–760M parameters and context lengths of 4k–16k tokens, ACP achieves a 2–3× attention speedup at zero performance cost — a case where the architectural forgetting mechanism and computational efficiency are the same property.*

---

> **Key Takeaway:** The forget gate creates emergent sparsity: local-context heads in FoX need not compute over their full window. ACP converts this into guaranteed-safe pruning, achieving a 2–3× attention speedup across 125M–760M parameter models. Longer contexts amplify the savings, inverting the usual cost penalty for long-context inference.

---

## §4 — Measure Gradients, Not Activations

Alongside the architectural advances, a diagnostic problem had been quietly undermining the entire plasticity intervention literature: the standard method for identifying dead neurons does not work in the architectures the field now uses most.

The τ-dormant neuron metric (Sokar et al., 2023) identifies dead neurons by activation magnitude — if a neuron's activation is consistently small across the training distribution, it is classified as dormant and a candidate for reinitialization. This approach works correctly for simple MLP architectures with ReLU activations, where activation magnitude and learning contribution are tightly correlated: a ReLU neuron that activates weakly also receives weak gradient.

In deeper, more complex architectures — residual networks, diffusion-based models, architectures with non-ReLU activations — the correlation breaks. Specifically, skip connections in residual networks create neurons that have high activation magnitude but contribute near-zero gradient signal. The activation path flows through the skip connection; the gradient path for the weight update is degenerate. τ-dormant, measuring only activation output, classifies these neurons as healthy. A gradient-based diagnostic correctly identifies them as learning-impaired.

Liu, Wu, Obando-Ceron, Castro, Courville, and Pan introduced GraMa (Gradient Magnitude Neural Activity Metric) to close this gap directly (Liu et al., arXiv:2505.24061). The architectural insight: in a residual network, the relevant quantity for a neuron's contribution to learning is not its activation value but the magnitude of the gradient flowing through it. A neuron with near-zero gradient does not update its weights — regardless of how much it activates. A neuron with large gradient learns meaningfully from every training step — regardless of its activation magnitude.

GraMa computes the gradient of the loss with respect to each neuron's pre-activation value and measures the magnitude of this gradient across a window of recent training steps. Low gradient magnitude → neuron is learning-impaired. The metric is architecture-agnostic: it makes no assumptions about activation functions, network connectivity, or skip connection topology.

ReGraMa extends GraMa into an intervention: reset neurons identified as learning-impaired by gradient statistics rather than activation statistics. In experiments across MuJoCo and the DeepMind Control Suite benchmarks, ReGraMa consistently improves learning performance compared to τ-dormant-guided reset (ReDo). The improvement is most pronounced on advanced vision RL agents — specifically, on the BRO-net architecture where τ-dormant-guided reset fails entirely to restore learning capacity while GraMa-guided reset succeeds. The method applies correctly to residual networks, diffusion models, and agents with varied activation functions.

The diagnostic implication extends to the entire plasticity measurement literature: if the field has been using τ-dormant to assess plasticity in residual networks, it has been systematically underestimating the extent of the dead neuron problem. Many neurons identified as "healthy" by activation metrics are contributing near-zero gradient and learning nothing. The plasticity crisis, measured by the correct diagnostic, is worse than activation-based assessments suggested.

(see Figure 3)

---

**Figure 3 — Why Activation Metrics Fail in Residual Networks**

```
  ══════════════════════════════════════════════════════════════════════════
   SIMPLE MLP (ReLU)                    DEEP RESIDUAL NETWORK
  ──────────────────────────────────────────────────────────────────────────

   x → [L1] → [L2] → [L3] → output     x → [L1] ──────────────────┐ (skip)
                                                ↓                   │
   Gradient path = activation path      [L2] → [L3] → + → output ←┘

   NEURON STATE     τ-dormant  GraMa    NEURON STATE      τ-dormant  GraMa
   ─────────────────────────────────    ──────────────────────────────────
   Low activation       ✗         ✗    Low act, low grad      ✗         ✗
   High activation      ✓         ✓    High act, low grad     ✓         ✗ ← MISSED
                                       Low act, high grad     ✗         ✓
   Metrics agree.                      High act, high grad    ✓         ✓

  ──────────────────────────────────────────────────────────────────────────
   "High act, low grad" neurons = skip-connection units that activate
   freely but whose weight updates contribute near-zero learning signal.
   τ-dormant classifies them as healthy.
   GraMa correctly identifies them as learning-impaired.

   Result: on advanced vision RL (BRO-net), τ-dormant-guided reset cannot
   restore learning capacity; GraMa-guided reset (ReGraMa) succeeds.
   Benchmarks: MuJoCo, DeepMind Control Suite.
  ══════════════════════════════════════════════════════════════════════════
```

*Figure 3: In simple MLP architectures, activation metrics (τ-dormant) and gradient metrics (GraMa) agree — both identify the same dead neurons. In residual networks — the architecture used in modern RL agents — skip connections create a class of high-activation, near-zero-gradient neurons that τ-dormant misses entirely. GraMa correctly identifies them as learning-impaired. The plasticity deficit in modern RL architectures is larger than activation-based diagnostics indicate.*

---

> **Key Takeaway:** The τ-dormant metric fails in residual networks because skip connections decouple activation magnitude from gradient contribution. GraMa (gradient magnitude) is the architecture-agnostic diagnostic that correctly identifies learning-impaired neurons across residual nets, diffusion models, and varied activation functions. ReGraMa, guided by GraMa, succeeds where τ-dormant-guided reset fails on advanced vision RL architectures.

---

## §5 — Plasticity Mitigation via Gradient Interference Reduction

Alongside the architectural work on FoX and the diagnostic work on GraMa, a third line from the Courville group attacks plasticity loss from a training-dynamics perspective — not through periodic reinit or gating, but by characterizing and reducing **churn**.

Churn, as defined by Tang, Obando-Ceron, Castro, Courville, and Berseth (2025), is network output variability for out-of-batch data induced by mini-batch gradient updates. When a gradient update reduces loss on the current mini-batch, it simultaneously changes predictions for inputs not in that batch. In a stable, high-plasticity network, churn is small — updates are specific and localized. In a plasticity-impaired network, churn is large — updates radiate destructively across the full input space.

The theoretical connection is formal: churn amplification is caused by NTK rank collapse. As the Neural Tangent Kernel matrix loses rank, gradient updates become increasingly low-dimensional and affect many off-batch inputs simultaneously. This is the mechanism behind plasticity loss viewed through a churn lens: **NTK rank collapse → churn amplification → gradient degradation → plasticity loss**. The link between the structural (NTK) and the behavioral (churn) levels is the diagnostic innovation.

Tang et al. demonstrate two complementary mechanisms by which reducing churn preserves plasticity (Tang et al., arXiv:2506.00592). First, a **gradient decorrelation effect**: churn reduction forces gradient updates to be more specific to the current batch, preventing the co-linear gradient collapse that drives rank loss. Second, a **step-size adjustment effect**: churn reduction adaptively scales down updates that would produce destructively large output changes on out-of-batch inputs, effectively acting as a learned, task-aware learning rate modulator.

Their intervention, C-CHAIN (Continual Churn Approximated Reduction), implements churn reduction for continual RL settings. C-CHAIN was tested across 24 continual RL environments spanning OpenAI Gym Control tasks, ProcGen procedurally generated environments, the DeepMind Control Suite, and MinAtar. Across these settings, C-CHAIN improves learning performance and outperforms baseline plasticity methods in most environments — with the performance advantage accumulating over the course of training, consistent with the mechanism: churn reduction prevents collapse from compounding.

The positioning of C-CHAIN in the programme is important. It is a training-procedure fix applied to standard architectures. It is the group's answer to "what can you do on the training side when architectural redesign is not an option?" — complementary to FoX's architectural prevention. The full solution requires both: architecture that does not accumulate plasticity debt (FoX), and training dynamics that do not generate that debt in the first place (C-CHAIN).

(see Figure 4)

---

**Figure 4 — The Churn–Plasticity Loop and C-CHAIN's Intervention**

```
  ═══════════════════════════════════════════════════════════════════════════
   THE PATHOLOGICAL LOOP                     C-CHAIN BREAKS IT
  ───────────────────────────────────────────────────────────────────────────

   Sequential task training               Sequential task training
          ↓                                        ↓
   Mini-batch gradient updates            Churn regularization applied
   overfit current batch                  at each gradient step
          ↓                                        ↓
   NTK matrix rank decreases             NTK rank maintained
          ↓                                        ↓
   Out-of-batch outputs become           Output variability for
   increasingly volatile (churn ↑)        off-batch inputs suppressed
          ↓                                        ↓
   Future gradient updates become        Gradient directions stay
   low-dimensional and co-linear          diverse across tasks
          ↓                                        ↓
   Effective learning rate falls         Plasticity maintained
   → PLASTICITY LOSS                     → C-CHAIN exits loop here

  ───────────────────────────────────────────────────────────────────────────
   Results (Tang et al., ICML 2025):
   24 continual RL environments: Gym Control · ProcGen · DMControl · MinAtar
   C-CHAIN outperforms baseline plasticity methods in most environments.
   Mechanism: gradient decorrelation effect + step-size adjustment effect
  ═══════════════════════════════════════════════════════════════════════════
```

*Figure 4: The churn–plasticity feedback loop as identified by Tang et al. (2025). NTK rank collapse amplifies output variability for out-of-batch data (churn), which degrades gradient quality, which accelerates rank collapse — a self-reinforcing cycle. C-CHAIN breaks this loop through explicit churn regularization, demonstrated across 24 continual RL environments.*

---

> **Key Takeaway:** Plasticity loss can be understood as a churn amplification cycle driven by NTK rank collapse. C-CHAIN (Tang et al., ICML 2025) breaks this cycle via gradient decorrelation and adaptive step-size adjustment across 24 diverse continual RL environments. This is the training-side complement to FoX's architectural prevention: together they address both the structural cause and the training dynamics that feed it.

---

## §6 — Scaling Depth: 1000 Layer Networks

The most provocative result in this section of the literature does not come from the Courville group. Wang, Javali, Bortkiewicz, Trzciński, and Eysenbach (Princeton / Warsaw University of Technology, NeurIPS 2025) asked a question the deep RL community had largely given up on: what if we simply go deeper?

The conventional wisdom — reinforced by multiple empirical studies — was that adding depth beyond 2–5 layers produces marginal gains or negative returns in RL. The RL training signal is sparse, targets are non-stationary, and high-variance updates destabilize the optimization landscape for very deep architectures. Prior work consistently reported limited benefit from depth scaling in RL.

Wang et al. demonstrate that this conventional wisdom is wrong for a specific, well-constructed setup. Using self-supervised RL (no explicit rewards — only goal-reaching objectives), combined with residual connections, layer normalization, and Swish activations, they scale network depth to 1024 layers and document consistent, dramatic performance improvements across locomotion and manipulation tasks (Wang et al., arXiv:2503.14858). Across simulated environments, their approach increases performance by 2×–50× compared to shallow baselines. Notably, performance does not scale smoothly — it jumps discontinuously at specific critical depths. For example, performance on Ant Big Maze jumps at 8 layers; on Humanoid U-Maze, the jump occurs at 64 layers. These discontinuous emergence points suggest qualitatively new behavioral capabilities at depth thresholds, analogous to emergent capabilities observed in large language models as parameter counts scale.

The empirical analysis reveals why depth helps: deeper networks exhibit improved stitching capabilities — the ability to compose learned sub-policies — learn more accurate value functions, and unlock the benefits of larger batch sizes. The architectural ingredients that make 1024-layer RL networks trainable (residual connections, careful normalization, Swish activations) are the same ingredients that prevent gradient vanishing and dead neuron accumulation in deep sequential training. Depth-based scaling and plasticity preservation are, in this sense, the same engineering problem viewed from opposite directions.

This paper does not resolve the continual learning plasticity crisis directly — it operates in a single-task self-supervised setting and does not address multi-task sequential training. But it establishes a foundational result: deep RL architectures are trainable when built correctly, and their depth confers qualitative capability improvements. The plasticity-preserving architectural principles are necessary conditions for reaching these depth scales.

(see Figure 5)

---

**Figure 5 — Scaling Depth in Self-Supervised RL**

```
  ══════════════════════════════════════════════════════════════════════════
   PERFORMANCE vs. NETWORK DEPTH (Wang et al., NeurIPS 2025 — schematic)
  ──────────────────────────────────────────────────────────────────────────

   Normalized  │
   Performance │
               │                                           ████ ████ ████
               │                                    ████ ████ ████ ████
               │                             ████ ████
   50%         ┤                      █ ████
               │               █ ████        ↑ emergent jump (task-dependent
   25%         ┤         ████ ██               depth threshold, e.g., 8L for
               │    ████ █                    Ant Big Maze, 64L for
    0%         ┤ ████                          Humanoid U-Maze)
               └───────────────────────────────────────────────────
                2L   4L   8L  16L  32L  64L  128L 256L 512L 1024L
                                 Network Depth

  ──────────────────────────────────────────────────────────────────────────
   Key findings:
   · 2×–50× performance improvement at maximum depth vs. 2-layer baseline
   · Performance jumps at critical depths (discontinuous, not smooth)
   · Deeper networks: improved stitching + more accurate value functions
                      + benefit from larger batch sizes
   · Architecture recipe: residual connections + layer norm + Swish activation
   · Setting: self-supervised goal-conditioned RL (no explicit rewards)
  ══════════════════════════════════════════════════════════════════════════
```

*Figure 5: Scaling depth from 2 to 1024 layers in self-supervised RL yields 2×–50× performance improvements across locomotion and manipulation tasks (Wang et al., NeurIPS 2025). Performance jumps at specific critical depth thresholds rather than scaling smoothly, suggesting emergent capabilities analogous to those in large language models. The architectural recipe that enables this depth — residual connections, layer normalization, Swish activations — is the same recipe that prevents plasticity collapse in deep sequential training.*

---

> **Key Takeaway:** Depth scaling in RL was long considered a diminishing return. Wang et al. (NeurIPS 2025) demonstrated empirically that networks up to 1024 layers achieve 2×–50× performance improvements, with emergent capability jumps at critical depth thresholds. The same architectural principles that make extreme depth trainable (residual connections, layer normalization) are those that preserve plasticity in continual settings.

---

## §7 — A Third Paradigm: Inference-Time Weight Editing

Training tricks address plasticity by changing how the network trains. Architectural changes address it by changing what the network is. Transformer² (Sun, Cetin, Tang, 2025, ICLR) introduces a third paradigm: change the network's effective weights at inference time, without any gradient steps.

The mathematical starting point is the singular value decomposition of weight matrices. Every weight matrix W ∈ ℝⁿˣᵐ in a Transformer can be decomposed as W = UΣVᵀ, where U ∈ ℝᵐˣʳ and V ∈ ℝⁿˣʳ are semi-orthogonal matrices and Σ ∈ ℝʳˣʳ is diagonal with singular values σ₁ ≥ σ₂ ≥ ... ≥ σᵣ. The linear operation can be decomposed as:

> y = W x = Σᵢ σᵢ uᵢ vᵢᵀ x

Each rank-1 component σᵢ uᵢ vᵢᵀ independently processes the input, contributing to the layer's output in proportion to σᵢ. This decomposition makes the modular structure of the weight matrix explicit.

Singular Value Fine-tuning (SVF), the core building block of Transformer², trains task-specific **expert vectors** z ∈ ℝʳ using reinforcement learning. Each component zᵢ of an expert vector encodes a learned, task-specific scaling factor for the corresponding rank-1 component. Transformer² trains K expert vectors z₁, ..., z_K, each corresponding to a different capability domain (coding, math, vision-language, general tasks, etc.). Critically, these expert vectors are not directly modifying the diagonal of Σ — they parameterize a learned, compositional mapping that produces task-specific projections of the SVD components. The mechanism is compact and RL-trained; it is not a lookup table, not a fixed rescaling, and not a simple linear combination of singular values.

During inference, Transformer² operates in two passes. **Pass 1 (dispatch)**: the model processes the input, observes hidden states, and identifies the task's domain using one of three dispatch strategies (prompt engineering, few-shot context identification, or learned dispatch). Based on this identification, a new expert vector z′ is computed as a mixture of z₁, ..., z_K. **Pass 2 (adaptation)**: the base model generates the actual response using weights modulated by z′ — with each rank-1 weight component σᵢ scaled by z′ᵢ. No gradient steps are taken. The base model is never modified.

The practical results: Transformer² consistently outperforms LoRA — the standard parameter-efficient fine-tuning baseline — while using fewer parameters. It generalizes across LLM architectures and across modalities, including vision-language tasks (Sun et al., 2025, ICLR). The source code is publicly available.

The continual learning interpretation: Transformer² does not prevent plasticity loss in the base model. Instead, it provides a mechanism to recover task-specific performance at inference time without touching training dynamics. A plasticity-impaired base model — one that has lost the capacity to learn new tasks via fine-tuning — can still produce task-appropriate outputs when its effective weights are modulated by the right z′ at inference time. This is adaptation without fine-tuning: the third paradigm in the solution space.

(see Figure 6)

---

**Figure 6 — Transformer² Two-Pass Adaptation Architecture**

```
  ══════════════════════════════════════════════════════════════════════════
   TRANSFORMER² INFERENCE PIPELINE (Sun et al., ICLR 2025)
  ──────────────────────────────────────────────────────────────────────────

   INPUT PROMPT ──────────────────────────────────────────────────────────→
                │
   ┌────────────▼─────────────────────────────────────────────────────────┐
   │  PASS 1: DISPATCH                                                     │
   │  Model processes prompt → observes hidden states                      │
   │  Dispatch strategy identifies task domain (e.g., [MATH])              │
   │  Compute: z' = α₁z₁ + α₂z₂ + ... + αₖzₖ                            │
   │  where z₁,...,zₖ are SVF-trained expert vectors (RL-trained)         │
   │  Note: z' is NOT a simple rescaling of Σ — it is a learned,          │
   │        compositional projection encoding task-specific patterns       │
   └──────────────────────────────────────────────────────────────────────┘
                          ↓
              z' vector computed (no grad step)
                          ↓
   ┌──────────────────────▼───────────────────────────────────────────────┐
   │  WEIGHT MODULATION via SVD COMPONENTS                                │
   │  For each weight matrix W = UΣVᵀ:                                   │
   │    Each rank-1 component σᵢ uᵢ vᵢᵀ scaled by z'ᵢ                   │
   │    → effective weights change per task without modifying base model  │
   └──────────────────────────────────────────────────────────────────────┘
                          ↓
   ┌──────────────────────▼───────────────────────────────────────────────┐
   │  PASS 2: ADAPTED RESPONSE                                            │
   │  Generate output using z'-modulated weights                          │
   │  NO GRADIENT STEPS · NO FINE-TUNING · BASE MODEL UNCHANGED          │
   └──────────────────────────────────────────────────────────────────────┘
                          ↓
   OUTPUT ─────────────────────────────────────────────────────────────→
  ──────────────────────────────────────────────────────────────────────────
   vs. LoRA: Transformer² outperforms LoRA, fewer parameters,
             generalizes across LLM architectures and modalities
  ══════════════════════════════════════════════════════════════════════════
```

*Figure 6: Transformer² (Sun et al., ICLR 2025) introduces inference-time weight editing via a two-pass mechanism. Pass 1 identifies the task and computes a mixed expert vector z′ from K SVF-trained, RL-optimized expert vectors. Pass 2 generates the response with base model weights modulated by z′ through the SVD component structure. The modulation is not a simple Σ rescaling — z′ encodes a learned, task-specific projection. No gradient steps are required; the base model is never modified during adaptation.*

---

> **Key Takeaway:** Transformer² is the third paradigm: inference-time weight editing without gradient steps. SVF trains task-specific expert vectors via RL over SVD components of weight matrices. At inference, a two-pass mechanism identifies the task and applies a z′-modulated weight structure — outperforming LoRA with fewer parameters, across architectures and modalities. Expert vectors are not a simple Σ rescaling; they encode learned, compositional task-specific projections.

---

## §8 — Architecture vs. Training vs. Inference-Time Editing: Which Wins?

The work described in this article can be organized into three tiers, each operating at a different level of the machine learning stack:

**Tier 1 — Training-procedure fixes.** Continual backpropagation (Dohare et al., 2024), spectral regularization, and C-CHAIN (Tang et al., 2025) apply to any architecture without structural modification. They compensate for plasticity failure modes: reinitializing dead neurons, preventing rank collapse, suppressing churn amplification. These are necessary in practice — most deployed systems use standard Transformers and cannot be redesigned — but they are fundamentally compensatory. The cause (a static-distribution architecture applied to a non-stationary problem) persists.

**Tier 2 — Architectural changes.** The Forgetting Transformer prevents plasticity debt from accumulating by giving the network an explicit information-decay mechanism at the level of attention. GraMa and ReGraMa provide the correct gradient-based diagnostic, revealing that the plasticity deficit in modern RL architectures is larger than activation-based metrics indicate. The 1000-layer result establishes that depth scaling is viable when paired with the right residual and normalization structure. These interventions address the cause: an architecture designed for the correct computational structure.

**Tier 3 — Inference-time editing.** Transformer² routes around plasticity loss rather than preventing or compensating for it. If a base model has lost the capacity to adapt to new tasks via fine-tuning, inference-time expert vector mixing can still produce task-appropriate outputs by modulating the effective weights at runtime. No gradient steps required.

None of these tiers is universally dominant. The right intervention depends on what the deployment scenario demands:

| Scenario | Best intervention |
|----------|------------------|
| High-frequency task switching, minimal gradient budget | Transformer² (zero gradient steps, immediate) |
| Long-horizon CRL where base model must generalize | FoX (architectural prevention is the durable fix) |
| Standard architecture in production, no redesign possible | C-CHAIN + continual backpropagation |
| Advanced vision RL with residual networks | GraMa / ReGraMa (correct dormancy diagnostic) |
| Goal of scaling depth in RL | Residual connections + layer norm + Swish (Wang et al.) |

The deeper insight: these three tiers are not competing — they are complementary. A maximally plasticity-preserving system would combine:
1. **Architecture**: FoX (forget gate prevents information accumulation from freezing)
2. **Training dynamics**: C-CHAIN (prevents NTK rank collapse from compounding)
3. **Diagnostic accuracy**: GraMa (identifies where intervention is actually needed)
4. **Inference flexibility**: Transformer² (rapid adaptation at deployment without touching the training pipeline)

This is the programme. It did not emerge from a single paper; it is a research arc built systematically by the Courville group at Mila, joined by collaborators across Princeton, Hong Kong, and Warsaw. Each paper addresses one failure mode. Together, they form the first complete engineering response to the plasticity crisis.

(see Figure 7)

---

**Figure 7 — The Architectural Plasticity Research Programme**

```
  ══════════════════════════════════════════════════════════════════════════
   THE COURVILLE-GROUP PLASTICITY PROGRAMME (2024–2025)
  ──────────────────────────────────────────────────────────────────────────

   FAILURE MODE                PAPER (venue)         MECHANISM
   ─────────────────────────────────────────────────────────────────────────
   [ARCHITECTURE]
   Attention retains           Forgetting             Forget gate f_ij:
   everything (no decay)       Transformer            data-dependent decay of
                               Lin et al.             unnormalized attn scores
                               ICLR 2025

   Forget-gate sparsity        Adaptive               Provably-safe pruning
   not exploited               Computation            threshold: ~70% FLOP
   computationally             Pruning (ACP)          reduction, 2–3× speedup
                               Lin et al.
                               COLM 2025

   [DIAGNOSTIC]
   τ-dormant misses dead       GraMa / ReGraMa        Gradient magnitude as
   neurons in ResNets          Liu et al.             architecture-agnostic
                               arXiv:2505.24061       dormancy metric

   [TRAINING DYNAMICS]
   NTK rank collapse →         C-CHAIN                Churn reduction:
   churn → plasticity loss     Tang et al.            gradient decorrelation
                               ICML 2025              + step-size adjustment

   [INFERENCE-TIME EDITING]
   Base model plasticity-      Transformer²           SVF expert vectors
   impaired for new tasks?     Sun et al.             (RL-trained) mixed
                               ICLR 2025              at inference time
  ══════════════════════════════════════════════════════════════════════════
```

*Figure 7: The architectural plasticity research programme as a coherent system. Each component addresses one failure mode of standard deep networks in continual settings. FoX targets the root structural cause; ACP converts the resulting sparsity into efficiency; GraMa provides the correct diagnostic; C-CHAIN addresses training dynamics; Transformer² provides inference-time adaptation. Together they form the first complete engineering response to the plasticity crisis.*

---

> **Key Takeaway:** The three paradigms — training-procedure fixes, architectural changes, and inference-time editing — are complementary, not competing. Each tier dominates in a different deployment scenario. A fully plasticity-preserving system combines FoX (architectural prevention) + C-CHAIN (training-dynamics stability) + GraMa (correct diagnostic) + Transformer² (inference flexibility). This is not a single paper — it is a coordinated research programme.

---

## § What Comes Next

This article established that architecture is the correct level of intervention for plasticity loss, and traced five papers that constitute the first coherent engineering programme for solving it. What does the rest of the series add?

**[→ A2: Does RL Teach LLMs to Reason?]** The Forgetting Transformer solves a structural stability problem. Article 2 addresses a different question: once the architecture is stable, does RL post-training actually produce new reasoning capability, or does it merely amplify what the base model already knows? Architectural stability is the necessary condition for asking this question cleanly — without it, RL post-training on a plasticity-impaired model produces noise, not signal. A6 is the prerequisite A2 assumes.

**[→ A7: Stable Deep RL at Scale]** §6 established that 1024-layer RL networks achieve dramatic capability improvements. Article 7 provides the stability toolkit — gradient clipping strategies, layer normalization schedules, training-step management — needed to reliably train them. The Forgetting Transformer scales to arbitrary depth without the special handling A7 describes, but A7's toolkit is necessary for stable deployment at the scale the 1000-layer result implies.

**[→ A11: Thinking Without Tokens: CTM]** The Forgetting Transformer's forget gate is per-position in the attention computation: it controls which sequence positions each head attends to. The Continuous-Time Model (CTM) in Article 11 goes further: per-neuron temporal memory updated at different timescales within a single forward pass, driven by internal recurrence rather than external sequence position. FoX is the first architectural step toward temporal heterogeneity in the attention stack; CTM is the next. [→ A11]

The core argument of this series is that the plasticity crisis is an architectural problem wearing a training problem's clothing. This article provides the first clean demonstration: one gate added to the attention mechanism converts the world's most widely deployed architecture into a continual learning substrate. The next articles investigate whether that substrate, once stable, can build genuine reasoning capability and sustain it at scale.

[← A4: The GVF framework anticipated the structural need for this separation — continual prediction targets as natural scaffolding for what to retain versus decay]

---

## Final Key Takeaways

1. **Architecture is the right level of intervention.** Standard softmax attention has no mechanism to decay old information — it was designed for static distributions. Training-procedure fixes compensate for this; the Forgetting Transformer eliminates it at the source.

2. **The forget gate is minimal and compatible.** FoX adds one data-dependent decay factor to unnormalized attention scores. It requires no positional embeddings, is compatible with FlashAttention, and outperforms the standard Transformer on long-context modeling and short-context downstream tasks.

3. **The forget gate creates exploitable computational sparsity.** Many FoX attention heads develop local structure through fast decay. Adaptive Computation Pruning (ACP) converts this into provably-safe pruning, achieving a 2–3× attention speedup across 125M–760M parameter models at zero performance cost.

4. **Activation-based dormancy metrics are insufficient for modern architectures.** In residual networks — the architecture used in modern RL — skip connections create high-activation, near-zero-gradient neurons that τ-dormant misses. GraMa (gradient magnitude) is the correct diagnostic; ReGraMa is the correct intervention.

5. **Churn is the training-dynamics fingerprint of plasticity loss.** NTK rank collapse amplifies output variability for out-of-batch data. C-CHAIN (Tang et al., ICML 2025) breaks this cycle via gradient decorrelation and adaptive step-size adjustment, demonstrated across 24 continual RL environments.

6. **Depth scaling in RL is viable and powerful.** With residual connections and proper normalization, networks up to 1024 layers yield 2×–50× performance improvements with emergent capability jumps at specific critical depth thresholds (Wang et al., NeurIPS 2025).

7. **Inference-time editing is the third paradigm.** Transformer² (Sun et al., ICLR 2025) trains task-specific expert vectors via RL over SVD weight components, then mixes them at inference time without gradient steps. This is not a simple Σ rescaling — expert vectors encode learned, compositional task-specific projections that outperform LoRA with fewer parameters.

8. **The three paradigms are complementary.** Training fixes apply to existing deployments. Architectural changes (FoX) provide durable prevention. Inference-time editing (Transformer²) enables rapid task adaptation without touching training. A maximally plasticity-preserving system combines all three.

---

## References

[1] Dohare, S., Hernandez-Garcia, J. F., Lan, Q., Rahman, P., Mahmood, A. R., & Sutton, R. S. (2024). **Loss of plasticity in deep continual learning.** *Nature*, 632, 768–774.

[2] Lin, Z., Nikishin, E., He, X. O., & Courville, A. (2025). **Forgetting Transformer: Softmax attention with a forget gate.** *International Conference on Learning Representations (ICLR 2025)*. arXiv:2503.02130.

[3] Lin, Z., Obando-Ceron, J., He, X. O., & Courville, A. (2025). **Adaptive computation pruning for the Forgetting Transformer.** *Conference on Language Modeling (COLM 2025)*. arXiv:2504.06949.

[4] Liu, J., Wu, Z., Obando-Ceron, J., Castro, P. S., Courville, A., & Pan, L. (2025). **Measure gradients, not activations! Enhancing neuronal activity in deep reinforcement learning.** arXiv:2505.24061.

[5] Tang, H., Obando-Ceron, J., Castro, P. S., Courville, A., & Berseth, G. (2025). **Mitigating plasticity loss in continual reinforcement learning by reducing churn.** *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*. arXiv:2506.00592.

[6] Wang, K., Javali, I., Bortkiewicz, M., Trzciński, T., & Eysenbach, B. (2025). **1000 layer networks for self-supervised RL: Scaling depth can enable new goal-reaching capabilities.** *Advances in Neural Information Processing Systems (NeurIPS 2025)*. arXiv:2503.14858.

[7] Sun, Q., Cetin, E., & Tang, Y. (2025). **Transformer-Squared: Self-adaptive LLMs.** *International Conference on Learning Representations (ICLR 2025)*. Sakana AI.
