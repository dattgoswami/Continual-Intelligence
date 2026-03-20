# GVFs as Proto-World-Models: The Alberta Plan Vindicated?

Article 4 of 12  | [CL][WM][RL] | Anchor papers: Sutton et al. 2011 (Horde) · Khetarpal et al. 2022 · Pan et al. 2025 · Hafner et al. 2023 · Micheli et al. ICLR 2023 | Series: Continual Intelligence

---

In 2011, a team at the University of Alberta published a paper about a robot learning to predict things about its own body. They called the architecture Horde. It appeared at an autonomous agents conference, not a machine learning venue. It was not widely cited by the researchers who would later build DreamerV3 or IRIS. By 2023, those world model systems had replicated — in latent space, using learned representations and multi-step imagination — a prediction structure that the Alberta group had outlined twelve years earlier in tabular feature form.

This article traces that lineage. Not to claim credit for the Alberta group: the gap between a 2011 robot predictor and a 2023 deep world model is large, and the Alberta plan missed crucial elements. But the *structure* of the prediction problem they were solving is the same structure that modern world models solve. The General Value Function formalism anticipated the decomposition that DreamerV3 implements. That is worth understanding precisely, because the correct decomposition — discovered in 2011 — is still the decomposition in 2023.

---

## §1 — What Is a General Value Function?

A standard value function asks one question: what is the expected cumulative reward from this state? Everything else — the policy, the reward function, the time horizon — is fixed. The value function answers a single predetermined question about the future.

A **General Value Function** (GVF) asks an arbitrary question about the future. Formally, a GVF is parameterized by four components (Sutton et al., 2011):

- a **cumulant function** ψ — the pseudo-reward, which can be *any* quantity observable from experience, not necessarily the task reward
- a **discount function** γ — controlling the time horizon over which the prediction accumulates
- a **termination function** β — defining when the prediction episode ends (the pseudo-episodic boundary)
- a **behavior policy** π — the policy under which the predictions are collected

With these four degrees of freedom, a GVF can answer questions like: "How far will I travel before I hit a wall?", "How hot will this motor get over the next five steps?", "How many steps until this subgoal is reached?" These are not reward-related questions. They are questions about the structure of the environment — its dynamics, its geometry, its temporal regularities.

The key realization: a collection of GVFs, each answering a different question about the future, constitutes a *prediction-based model of the environment*. Not a generative model. Not a latent dynamics model in the DreamerV3 sense. But a *factored, question-answer model* of what the world does next under different conditions.

The Alberta group named the goal **knowledge as predictions** — a semantic framework in which knowledge is defined as accurate future prediction. Sutton et al. (2011) state this explicitly: "A value function asks a question — what will the cumulative future reward be? — and an approximate value function provides an answer to that question. The approximate value function is the knowledge, and its match to the value function — to the actual future reward — defines what it means for the knowledge to be accurate." This grounding makes GVF-based knowledge falsifiable: you can check whether the predictions are correct. Standard symbolic knowledge representations do not have this property.

> **Key Takeaway:** A General Value Function generalizes the standard value function by replacing the fixed reward signal with an arbitrary cumulant function ψ, parameterized also by a termination condition β and a discount γ. A collection of GVFs constitutes a proto-world-model: a factored, verifiable representation of environment dynamics. The structure (ψ, γ, β, π) maps directly to what modern world models compute.

---

## §2 — Horde: Massively Parallel Prediction as World Modelling

The Horde architecture (Sutton et al., 2011) implements the GVF idea at scale. The design is deliberately modular: a large number of independent reinforcement learning sub-agents, called **demons**, each responsible for answering exactly one predictive or goal-oriented question about the world. Each demon is a GVF instantiation — its own policy, its own cumulant, its own termination condition, unrelated to those of any other demon.

Four properties of Horde are architecturally significant.

**Off-policy learning from shared experience**: Demons update from data collected by the system's actual behavior policy, regardless of the specific demon's own policy. Whatever the robot does generates training data for all demons simultaneously. The computational cost of running many demons is shared across the data stream.

**Constant time and memory per step**: Horde "runs in constant time and memory per time step, and is thus suitable for learning in real time" (Sutton et al., 2011). As more demons are added, the per-step cost scales linearly with demon count but remains bounded — making the architecture suitable for lifelong deployment.

**Gradient-based TD learning with function approximation**: Gradient-based temporal-difference learning is used throughout, allowing scaling to large feature spaces without tabular state enumeration.

**Factored, modular knowledge**: Because each demon answers a single question, the knowledge is structured. You know what each demon predicts, and you can evaluate individual demons against ground truth. The knowledge is decomposed, not entangled.

What does this look like as a world model? Horde does not generate trajectories. It does not learn a latent space. But it predicts, in parallel, a broad set of environmental properties — and those predictions collectively describe how the world behaves under various conditions. The demon that predicts "how hot will the motor get?" is modeling a specific environmental dynamic. The demon that predicts "how long before I reach this landmark?" is modeling spatial geometry. These are world-modeling computations in prediction form.

The structural insight: modern world models learn to predict **many things simultaneously** — observations, rewards, values, episode termination — inside a learned latent space. Horde was the first practical architecture for massively parallel prediction. It operated in feature space rather than latent space. That is the primary technical gap. But the prediction structure is the same.

What Horde missed — and this is essential for intellectual honesty — is the **grounding problem**. Demons answer questions that humans specify. The cumulant function ψ must be defined by the designer. The demons are not discovering what to predict; they are executing pre-specified predictions. Modern world models learn both the *latent representation* and the *prediction targets* jointly, through backpropagation. The Alberta plan's GVFs are grounded in experience — the match to future observations defines correctness — but the questions themselves are not learned. This is the gap that DreamerV3 closes.

---

**Figure 1 — The Alberta Lineage: From GVFs to Modern World Models**

```
YEAR    PAPER / CONTRIBUTION                    DOMAIN      KEY CAPABILITY
─────────────────────────────────────────────────────────────────────────────────
2011  ●──────────────────────────────────────●  [CL/WM/RL]
      Horde (Sutton et al.)                      Massively parallel GVF prediction;
      University of Alberta + McGill             off-policy demons; constant-time
                                                 architecture; hand-specified ψ
      │
      │  "structural lineage"
      │
2017– ●──────────────────────────────────────●  [CL/RL]
2018  Eigenoptions (Machado & Bowling 2016;      Options discovered via successor
      Machado et al. 2017)                       representation eigenvectors;
      University of Alberta                      structured exploration
      │
      │
2021  ●──────────────────────────────────────●  [CL/WM/RL]
      GVF Networks (Schlegel et al.)             GVFs for state construction in
      JAIR 70 (2021) 497–543                     partially observable environments;
      University of Alberta                      multi-step prediction as
                                                 representation learning
      │
      │                          ┌────────────── [WM/RL]
2022  ●──────────────────────────│──────────────● TD-MPC (Hansen et al., ICML 2022)
      │                          │                Latent MPC + TD value function;
      │                          │                task-oriented dynamics model
      │                          │
2023  ●──────────────────────────│──────────────● [WM/RL]
      Modern endpoints            │               DreamerV3 (Hafner et al.)
                                 │               RSSM; 150+ domains; single config
                                 └──────────────  IRIS (Micheli et al., ICLR 2023)
                                                 Transformer WM; >1.0 HNS Atari 100k
      │
      │
2025  ●──────────────────────────────────────●  [CL/WM/RL]
      Life-Long Visual WM (Pan et al.)           First visual CRL realization of
      Option Basis / Laplacian Keyboard           GVF lineage; mixture-of-Gaussians
      (Chandrasekar & Machado, RL Journal 2025)  task dynamics; generative replay
      Value-Aware Eigenoptions
      (Kotamreddy & Machado, RLC 2025)
─────────────────────────────────────────────────────────────────────────────────
Key: ● = significant paper    │ = structural lineage (not direct citation)
     [CL] Continual Learning  [WM] World Models  [RL] Reinforcement Learning
─────────────────────────────────────────────────────────────────────────────────
```

*Figure 1: The Alberta lineage from GVFs to modern world models, 2011–2025. Each node represents a significant paper; the connecting lines show structural lineage, not direct citation chains. The correspondence between 2011 Horde's prediction demons and 2023 DreamerV3's latent dynamics components reflects a common underlying decomposition: arbitrary prediction targets, discount-weighted horizons, episode termination signals, and off-policy data reuse.*

---

> **Key Takeaway:** Horde runs constant-time, off-policy, massively parallel prediction — the computational structure of a modern world model, implemented in feature space. Its critical limitation: cumulant functions are hand-specified by designers, not learned from data. The questions the demons answer are correct in form; the mechanism for discovering which questions to ask is missing.

---

## §3 — General Value Functions as a Basis for Continual Learning

The Horde paper introduced GVFs as a knowledge representation. The formalization of *how GVFs enable representation learning for continual adaptation* came a decade later.

Schlegel et al. (2021) published in the *Journal of Artificial Intelligence Research* a paper on **General Value Function Networks** — a framework in which GVF predictions are used to construct state representations for partially observable environments. The core insight: multi-step predictions constrain a recurrent network to encode temporal structure. If a recurrent network is trained to predict GVF targets at multiple horizons, it must build an internal state that retains the information necessary to make those predictions accurately.

The key problem this solves is state construction under partial observability. Standard RNN training via truncated backpropagation through time (BPTT) is sensitive to the truncation window — short truncation misses long-range dependencies; long truncation is computationally expensive and unstable. GVF predictions provide an alternative supervision signal: by asking the network to predict specific future quantities, the designer injects temporal structure that guides the RNN toward useful representations, even when BPTT is truncated.

The paper explores whether "conditioning a RNN on its own predictions" improves state construction — a question at the intersection of prediction-based representation learning and continual adaptation. GVF auxiliary predictions, if their cumulants track stable environmental dynamics, can anchor the representation across task changes. The predictions serve as a regularizer that prevents the representation from collapsing into a task-specific solution.

[← A1: The Plasticity Crisis] identified dead neurons and rank collapse as the mechanistic causes of plasticity loss. GVF-based auxiliary prediction addresses the same underlying issue through a different route: representation anchoring via stable prediction targets. If the network is continuously supervised on prediction targets that do not change with each task, gradient flow to the representation layers is maintained even when the task-specific head is being updated. This is a complementary mechanism — discovered by the Alberta group years before the plasticity literature named the problem.

> **Key Takeaway:** GVF Networks (Schlegel et al., 2021) extend Horde from prediction to representation: by training a recurrent network on multi-step GVF targets, the network is implicitly regularized to encode stable environmental structure. This addresses the plasticity loss problem from a representation-stability angle rather than a gradient-diversity angle — a complementary, earlier-developed mechanism that the plasticity literature has not widely credited.

---

## §4 — Options as Temporal Abstractions for CRL

The GVF framework handles predictions at a single timescale. But real environments have **hierarchical temporal structure** — step-level dynamics, room-level subgoals, task-level plans. Representing all of these at one timescale is both computationally wasteful and representationally insufficient.

The **options framework** (Sutton et al., 1999) addresses this by introducing temporally extended actions — behaviors that last multiple steps, with their own internal policies and termination conditions. An option is a triple (I, π, β): an initiation set I (where the option can be started), a policy π (the option's behavior), and a termination condition β (when the option ends). The agent chooses options, executes them for multiple steps, and learns a high-level policy over options.

Options are the temporal analogue of GVFs. Just as a GVF generalizes the value function by making the cumulant arbitrary, an option generalizes the action by making the behavior temporally extended. Together, GVFs and options form a two-dimensional generalization of standard RL: arbitrary prediction targets (GVF dimension) × arbitrary temporal scope (options dimension).

Klissarov et al. (2025) survey the full landscape of hierarchical RL, situating options as the foundational temporal-abstraction mechanism for agents that must "flexibly reason over different timescales by developing agents that learn, predict, and act in the world at multiple levels of abstraction." For CRL specifically, well-chosen options can span task changes — options that represent subgoals or skills useful across multiple tasks provide a behavioral scaffold analogous to the world model's representational scaffold. [→ A6: The Forgetting Transformer extends this structural separation into the attention mechanism itself — addressing the same decomposition need at the architecture level.]

**How do you discover which options to use?** Chandrasekar & Machado (2025) approach this through the graph Laplacian. The eigenvectors of the environment's graph Laplacian encode the environment's geometric structure — they describe how states relate to each other in terms of connectivity, not reward. Options derived from these eigenvectors (**eigenoptions**) are not task-specific; they describe the environment's topology, which is stable across task changes.

The **Option Keyboard** framework (Chandrasekar & Machado, 2025) uses eigenoptions as a **basis**: a compact set of options from which any behavior in the environment can be composed. The **Laplacian Keyboard** reduces this basis further, requiring a substantially smaller set of eigenoptions while matching the performance of the full basis. The key empirical result: "a sufficiently large eigenoption basis, combined with Generalized Policy Improvement, can recover near-optimal policies in the goal-reaching tasks we considered" (Chandrasekar & Machado, 2025). This is a proto-world-model result: the environment's structure, captured in the Laplacian's eigenvectors, is sufficient to generate efficient behaviors for any reward function without task-specific retraining.

> **Key Takeaway:** Options extend GVFs into temporal abstractions — behaviors that span multiple steps and represent persistent subgoals. Eigenoptions, derived from the graph Laplacian's eigenvectors, provide a task-agnostic option basis that encodes the environment's geometric structure. The Laplacian Keyboard (Chandrasekar & Machado, 2025) demonstrates that a compact eigenoption basis supports near-optimal policies for arbitrary reward functions — a geometric approach to the world-modeling problem.

---

## §5 — Eigenoptions: Spectral World Representation

The connection between eigenoptions and world models deepens when you examine the **successor representation** from which eigenoptions are derived.

The successor representation (SR), described by Kotamreddy & Machado (2025), "encodes the 'temporal distance' between states. It represents each state s as an |S| dimensional vector that contains the expected discounted visitation from s to every state s′." This is a world model in a specific sense: the SR captures transition dynamics under a policy, not as a generative model, but as a predictive representation of reachability. It tells you where you can get to, and how quickly, from any starting state.

Eigenoptions are discovered from the eigenvectors of the SR matrix (or equivalently, the graph Laplacian under a random walk). The eigenvectors with small eigenvalues correspond to slow-changing, globally smooth features of the state space — directions along which neighboring states vary slowly. Options that navigate along these eigenvectors discover global spatial structure. As Kotamreddy & Machado (2025) note, citing the original eigenoptions work (Machado & Bowling, 2016; Machado et al., 2017): eigenoptions originally addressed exploration challenges and have since been shown to scale to high-dimensional problems.

Kotamreddy & Machado (2025) investigate whether eigenoptions, beyond exploration, can accelerate **credit assignment** in model-free RL. The findings are nuanced but instructive for the CRL lineage:

- **Pre-specified eigenoptions** aid not only exploration but also credit assignment. When eigenoptions are provided to the agent in advance — rather than discovered online — the agent benefits from temporal abstraction for learning as well as for exploration.

- **Online discovery** is fragile: discovering eigenoptions simultaneously while using them for exploration can bias the agent's experience. "Early inaccuracies in option learning can skew the agent's experience, limiting its ability to fully explore the environment, thereby hindering credit assignment" (Kotamreddy & Machado, 2025). The agent gets stuck in a suboptimal exploration regime of its own making.

- **Deep RL with non-linear function approximation** introduces additional complexity: termination conditions become harder to define accurately when the option-value function is approximated, and approximation errors compound across the option's duration.

This tension — pre-specified structure that works robustly versus learned structure that is flexible but fragile — is precisely the tension that separates Horde from DreamerV3. Horde's cumulants are pre-specified; they work when chosen correctly. DreamerV3's representations are learned jointly; they are more flexible but require stable training signals to converge. The Alberta group's 2025 work is rediscovering, from the options direction, a problem the world model community encountered from the representation direction.

> **Key Takeaway:** Eigenoptions are spectral world representations — they capture the environment's transition geometry independent of any specific reward. Pre-specified eigenoptions aid both exploration and credit assignment (Kotamreddy & Machado, 2025). Online eigenoption discovery is fragile, mirroring the instability of learned GVF cumulants. The fundamental tension is between pre-specified structure (robust, limited) and learned structure (flexible, fragile) — the same tension that GVFs faced in 2011.

---

## §6 — From GVFs to Visual World Models (2025)

The most direct modern realization of the GVF lineage in a CRL setting is the **Life-Long World Model** (Pan et al., 2025). The paper's framing makes the connection explicit: "an ideal world model can provide a non-forgetting environment simulator, which enables the agent to optimize the policy in a multi-task learning manner based on the imagined trajectories from the world model" (Pan et al., 2025). This is the GVF hypothesis restated in modern terms. Horde said: predict many things about the world in parallel, and those predictions constitute world knowledge. Pan et al. say: learn a world model that generates imagined trajectories, and use those trajectories to avoid forgetting across tasks.

The Life-Long World Model's technical innovations close the two gaps that Horde left open:

**Task-specific latent dynamics with catastrophic forgetting prevention**: A mixture of Gaussians models task-specific latent dynamics, allowing the model to maintain separate trajectory distributions for different tasks without overwriting them. This is the forgetting-prevention mechanism that Horde's demons lacked — Horde's off-policy updates could overwrite prior demon predictions as the data distribution shifted.

**Generative experience replay**: Rather than storing past transitions (expensive, and impossible for continuous CRL), the model generates synthetic past experiences from the world model itself. This is the memory mechanism that GVF-based systems lacked entirely. Demons could not revisit past states without a generative replay mechanism; the Life-Long World Model generates those states on demand.

An additional innovation addresses a downstream problem specific to CRL: **exploratory-conservative behavior learning** handles the value estimation challenge for previous tasks — ensuring that the policy learned on imagined past trajectories does not collapse to a degenerate solution for tasks no longer actively trained on.

The paper evaluates on DeepMind Control Suite and MetaWorld, demonstrating that the Life-Long World Model "remarkably outperforms the straightforward combinations of existing continual learning and visual RL algorithms" (Pan et al., 2025). The non-forgetting imagined simulator is doing the work that Horde's factored prediction did in tabular feature form — but now in latent space, with visual observations, and with an explicit mechanism for protecting past knowledge.

Three frontier world models define the endpoint that the GVF lineage was building toward (see Figure 2):

**DreamerV3** (Hafner et al., 2023): A Recurrent State Space Model (RSSM) trained to predict future observations, rewards, and episode termination in latent space. The policy and value function are trained *entirely on imagined model rollouts* — the agent never updates directly from real environment interaction during policy training. This multi-step imagination is the capability Horde most conspicuously lacked. DreamerV3 applies a single fixed hyperparameter configuration across more than 150 diverse tasks spanning Atari, ProcGen, DMLab, Minecraft, and DMControl.

**IRIS** (Micheli et al., 2023): A discrete autoencoder compresses visual observations into categorical latent tokens; an autoregressive Transformer then models sequences of those tokens as the world model. Policy training occurs inside this Transformer world model. With only 100k environment interactions — strictly limited — IRIS achieves a mean human-normalized score exceeding 1.0 on the Atari 100k benchmark, surpassing human performance on 10 of 26 games tested (Micheli et al., 2023). The Transformer's attention mechanism provides the long-range temporal conditioning that GVF networks approximated with multi-step prediction targets.

**TD-MPC** (Hansen et al., 2022): A latent task-oriented dynamics model trained jointly with a terminal value function via temporal-difference learning. The world model is used for short-horizon trajectory optimization; the value function handles long-horizon return estimation without full model rollouts. This split — model for local planning, value function for global return — is the architectural descendant of the GVF design: separate prediction targets at different time horizons, combined through a learned value function.

---

**Figure 2 — GVF Formalism → DreamerV3: The Structural Correspondence**

```
═══════════════════════════════════════════════════════════════════════════════
   GVF FORMALISM (Horde, 2011)            DREAMERV3 (Hafner et al., 2023)
───────────────────────────────────────────────────────────────────────────────

   Cumulant function ψ(s, a, s′)    ──►   Reward model r̂(z_t, a_t)
   (arbitrary pseudo-reward;               (learned reward predictor in latent
    any observable quantity)                space; any quantity of interest)

   Discount function γ(s)           ──►   Discount factor γ ∈ (0, 1)
   (state-dependent horizon;               (latent trajectory rollout horizon;
    fade prediction over time)             controls how far to imagine)

   Termination function β(s)        ──►   Continuation predictor ĉ(z_t, a_t)
   (pseudo-episodic boundary;              (episode termination signal;
    when to end the prediction)            predicts when imagined episode ends)

   GVF prediction target            ──►   Latent state prediction ẑ_{t+1}
   (expected cumulative ψ                  (RSSM latent dynamics p(z_t | h_t);
    under discount and β)                  world model's core prediction)

   Behavior policy π                ──►   World model imagination rollouts
   (policy under which GVF                 (policy trained on model-generated
    is evaluated off-policy)               trajectories; behavior follows WM)

   Off-policy TD learning            ──►   Model-based RL in imagination
   (gradient-based GVF update              (backprop through imagined episodes;
    from shared experience)                 actor/critic trained on WM rollouts)

───────────────────────────────────────────────────────────────────────────────
   WHAT CHANGED (2011 → 2023):
   · Feature space → learned latent representations
   · Hand-specified ψ → jointly learned prediction targets
   · No imagination → multi-step imagined rollouts
   · Independent demons in parallel → unified RSSM with shared latent state

   WHAT IS PRESERVED:
   · Factored prediction decomposition: target / horizon / termination / policy
   · Discount-weighted future prediction
   · Episode boundaries as first-class prediction signal
   · Off-policy data reuse across prediction heads
═══════════════════════════════════════════════════════════════════════════════
```

*Figure 2: Structural correspondence between GVF formalism (Horde, 2011) and DreamerV3 (Hafner et al., 2023). Each row maps a GVF component to its modern equivalent. The representation changed substantially (feature space to latent space); the prediction decomposition did not. The Alberta plan correctly identified the (ψ, γ, β, π) structure in 2011. What changed from 2011 to 2023 is where the computation happens and whether the prediction targets are pre-specified or learned.*

---

> **Key Takeaway:** The Life-Long World Model (Pan et al., 2025) is the first CRL system to directly close Horde's two critical gaps: catastrophic forgetting of prior task dynamics (via mixture-of-Gaussians latent space + generative replay) and the absence of multi-step planning (via imagined trajectory optimization). The GVF → DreamerV3 structural correspondence is one-to-one at the component level: each GVF parameter has a direct modern equivalent. The level of abstraction changed; the decomposition did not.

---

## §7 — The CRL Desiderata Test: Does GVF-WM Pass?

Khetarpal, Riemer, Rish & Precup (2022) provide the most complete formal taxonomy of what a continual reinforcement learner must do. Published in the *Journal of Artificial Intelligence Research*, the paper defines seven desiderata for a continual learner: (1) learn online; (2) learn generalizable behaviors and skills while solving tasks; (3) operate task-agnostically across changes; (4) learn incrementally with no fixed training set; (5) build behaviors that can be composed into richer skills; (6) retain previously learned abilities without catastrophic interference; and (7) adapt efficiently to changes and recover quickly (Khetarpal et al., 2022).

Applying these desiderata systematically across the GVF lineage reveals both the progress made over fourteen years and the work that remains.

---

**Figure 3 — CRL Desiderata Satisfaction: GVF Lineage vs. Modern World Models**

```
═══════════════════════════════════════════════════════════════════════════════════
                                         Horde    Life-Long   DreamerV3     IRIS
DESIDERATA (Khetarpal et al. 2022)       (2011)   WM (2025)   (2023)        (2023)
───────────────────────────────────────────────────────────────────────────────────
1. Learn online                          ✅        ✅          ✅            ✅
   (real-time, no offline phase)

2. Learn generalizable behaviors         ⚠️        ✅          ✅            ✅
   (transferable across tasks)           demons    imagined    single-config  context-
                                         are task- planning    transfers      dependent
                                         specific  transfers   across 150+    planning

3. Task-agnostic                         ❌        ⚠️          ✅            ⚠️
   (no task ID required)                 cumulants task labels no task        game-
                                         hand-spec needed      labels         specific
                                                                              WM

4. Incremental learning                  ✅        ✅          ✅            ✅
   (no fixed training set)

5. Compositional skills                  ⚠️        ⚠️          ⚠️            ⚠️
   (behaviors build on behaviors)        options   limited     limited        limited
                                         partial

6. Retain prior abilities                ❌        ✅          ⚠️            ⚠️
   (minimize catastrophic forgetting)    no        generative  no explicit    no explicit
                                         mechanism replay +    anti-forget    anti-forget
                                                   MoG latents mechanism      mechanism

7. Adapt efficiently to changes          ⚠️        ✅          ✅            ✅
   (recover quickly per task)            per-demon exploratory single-step    imagination-
                                         adaptation conserv.   planning       based
                                                   BL                         planning
───────────────────────────────────────────────────────────────────────────────────
   Desiderata satisfied (approx.)       2/7        5–6/7       5–6/7         4–5/7
───────────────────────────────────────────────────────────────────────────────────
Key: ✅ satisfies  ⚠️ partially satisfies  ❌ fails
Ratings are qualitative assessments based on paper claims, not formal proofs.
═══════════════════════════════════════════════════════════════════════════════════
```

*Figure 3: CRL desiderata satisfaction across the GVF–world-model lineage. Horde (2/7) fails at desiderata 3 (task-agnosticity, because cumulants are hand-specified per task) and 6 (catastrophic forgetting prevention, absent from the architecture). The Life-Long World Model (5–6/7) closes both gaps via generative replay and task-specific latent dynamics. Desideratum 5 (compositional skills) remains partially open across all lineage systems; the options and eigenoptions research track is the active Alberta-group direction targeting it.*

---

The desiderata analysis reveals a specific failure pattern for Horde: **desiderata 3 and 6** are its two structural failures. Hand-specified cumulants require task-specific design effort (desideratum 3 violation), and the off-policy TD updates across shared experience have no mechanism for protecting predictions learned on prior tasks from being overwritten by new data (desideratum 6 violation).

The Life-Long World Model directly addresses both. Generative replay handles desideratum 6: by replaying synthetic past trajectories alongside new task data, the model retains prior knowledge without storing real past transitions. Task-specific latent dynamics with mixture-of-Gaussians handles the representation-level separation that prevents task distributions from interfering with each other.

Desideratum 5 — compositional skills, behaviors that build on behaviors — remains partially open across the entire lineage, including modern world models. This is precisely the gap that the options and eigenoptions work addresses. The Laplacian Keyboard (Chandrasekar & Machado, 2025) is the most direct attempt to satisfy desideratum 5 within the GVF lineage: a compact eigenoption basis from which arbitrary behaviors can be composed across task changes.

> **Key Takeaway:** Horde satisfies only 2 of 7 CRL desiderata. The Life-Long World Model (Pan et al., 2025) satisfies 5–6 of 7 — directly closing the forgetting and task-agnosticity gaps that GVF-based approaches left open. Compositional skills (desideratum 5) remain partially open across all lineage systems. The eigenoptions track is the active research direction targeting this remaining gap.

---

## §8 — Verdict: What the Alberta Plan Got Right and What It Missed

**What it got right:**

The GVF formalism correctly identified the **fundamental decomposition** of world knowledge in RL: a prediction parameterized by a target quantity (ψ), a time horizon (γ), and a termination signal (β), evaluated under a behavior policy (π). Every modern world model implements this structure. DreamerV3 predicts rewards, continuation signals, and future latent states — all under its learned RSSM dynamics. IRIS predicts token sequences. TD-MPC predicts task-relevant latent dynamics and terminal values. The components are the same as GVF's four-parameter tuple, reparameterized in latent space.

The Alberta plan also correctly identified **off-policy learning from shared experience** as the scalable mechanism for massively parallel prediction. Horde's demons share the robot's collected experience; DreamerV3 trains all prediction heads — observation, reward, continuation — from the same latent trajectory rollouts. The computational pattern is preserved.

Finally, the options and eigenoptions work correctly identified that **temporal abstraction is necessary** for agents operating across task sequences. The eigenoption basis provides task-agnostic structure derived from the environment's topology. Klissarov et al. (2025) confirm that HRL's core benefits — better exploration, faster learning, and cross-task generalization — are precisely the capabilities that CRL agents need to sustain across changing task distributions.

**What it missed:**

The grounding problem is the most fundamental gap. GVF cumulants must be specified by designers. The system answers predetermined questions, not the questions the environment makes important. Modern world models learn their prediction targets jointly with their representations, via gradient descent through imagined trajectories. DreamerV3 does not have a designer specifying what to predict — it learns a latent space in which prediction is maximally informative, with no hand-specified cumulants.

Second, Horde has no **multi-step imagination** capability. GVFs predict the expected return under a policy from a fixed start state, but they do not generate imagined future state sequences. Modern world models use their learned dynamics for planning: imagining sequences of future states, selecting actions based on imagined returns, and training policies entirely inside the imagined model. This multi-step rollout capability is what gives DreamerV3 and IRIS their sample efficiency advantages, and it is structurally absent from the GVF framework.

Third, the Alberta plan predates **attention mechanisms**. GVF networks' temporal conditioning is fixed-weight discounting — γ determines how much future predictions discount. Modern world models, particularly IRIS, use Transformer attention to selectively attend to relevant past observations across arbitrarily long contexts. This enables state conditioning far richer than fixed-discount aggregation.

The honest framing: the Alberta plan's **structure** was ahead of its time. Its **mechanism** was not. A prediction-based decomposition of world knowledge, with factored prediction targets, off-policy learning, and temporal abstraction — these design principles are correct and are implemented in every major modern world model. The mechanism for discovering *what to predict*, *how to represent it*, and *how to use it for planning* required another fourteen years to develop. The questions were right; the tools to answer them did not yet exist.

> **Key Takeaway:** The Alberta plan was structurally vindicated, mechanistically superseded. The prediction decomposition (ψ, γ, β, π) was correct in 2011 and remains the decomposition in 2023. The missing mechanisms — joint representation learning, multi-step imagination, attention — took until 2023 to develop. "Ahead of its time" means the structure was right before the technology existed to implement it. That is the correct framing; it does not mean the Alberta plan was complete.

---

## § What Comes Next

This article traced the Alberta lineage: from GVF predictions (Horde, 2011) through temporal abstractions (eigenoptions, option bases) through GVF-based representation learning (Schlegel et al., 2021) to the modern visual CRL realization (Pan et al., 2025) and three frontier world model endpoints (DreamerV3, IRIS, TD-MPC).

**[← A3: The Big World Hypothesis]** established the theoretical necessity of world models: the environment is irreducibly larger than any agent, so the agent must maintain a local predictive model continuously. The Alberta plan is the engineering response to that necessity, built before the theory was formally stated. The GVF structure that Sutton and colleagues arrived at empirically in 2011 is the structure that Lewandowski et al.'s computational embedding result explains mathematically in 2023.

**[→ A6: The Forgetting Transformer]** shows what happens when the world model insight is applied at the architecture level. The GVF lineage addressed CRL through prediction decomposition — what to predict, and how to aggregate it. The Forgetting Transformer addresses it through attention modification — how the network's capacity to continue predicting is structurally maintained. The approaches are complementary: GVFs define the prediction problem; the Forgetting Transformer defines the architectural conditions under which prediction can be sustained indefinitely. [→ A6]

**[→ A9: Frontier Reasoning Systems]** traces how frontier systems — DeepSeek-R1, ProRL, MiniMax-M1 — implicitly inherit the GVF lineage's structural insight without acknowledging it. Every frontier system that trains simultaneous reward models, value functions, and continuation predictors is running a descendant of Horde's parallel prediction architecture. The Alberta plan's structural insight has become an industry default; its origin is forgotten.

The vindication question in this article's title — "The Alberta Plan Vindicated?" — has a precise answer. **Structurally: yes.** The prediction decomposition that Horde instantiated in 2011 is the prediction decomposition that DreamerV3 implements in 2023. That is not coincidental; it is correct problem decomposition. **Mechanically: no.** The Alberta group did not anticipate latent-space learning, multi-step imagination, or self-supervised representation learning. The structure was right; the technology to realize it took a decade to arrive.

That is the correct meaning of "ahead of its time."

---

> **Final Key Takeaways**
>
> 1. **A General Value Function is a proto-world-model.** GVF tuple (ψ, γ, β, π) predicts arbitrary aspects of experience under a behavior policy — not just reward. A collection of GVFs answering different questions about the future constitutes factored, verifiable world knowledge. The decomposition is correct.
>
> 2. **Horde (2011) is the first massively parallel prediction architecture.** Off-policy demons running in constant time and memory, with gradient-based TD learning — the computational structure of a modern world model, implemented in feature space. Critical gap: hand-specified cumulants, no multi-step imagination, no representation learning.
>
> 3. **GVF Networks (2021) extended the framework to representation learning.** Multi-step GVF predictions constrain RNNs to encode stable environmental structure — a prediction-based approach to the plasticity problem that the gradient-diversity literature addresses differently. Earlier-developed, less-cited.
>
> 4. **Eigenoptions bring environment topology into temporal abstraction.** The Laplacian Keyboard (Chandrasekar & Machado, 2025) demonstrates that a compact eigenoption basis recovers near-optimal policies for arbitrary reward functions. Pre-specified eigenoptions aid credit assignment as well as exploration (Kotamreddy & Machado, 2025); online discovery remains fragile.
>
> 5. **The Life-Long Visual World Model (Pan et al., 2025) closes Horde's two critical gaps.** Mixture-of-Gaussians latent dynamics for non-forgetting task separation, plus generative experience replay for behavioral retention. Satisfies 5–6 of 7 CRL desiderata; Horde satisfies 2.
>
> 6. **GVF → DreamerV3 structural correspondence is one-to-one.** Cumulant ψ → reward model. Discount γ → rollout horizon. Termination β → continuation predictor. Off-policy TD → model-based RL in imagination. The representation level changed (features → latents); the decomposition did not.
>
> 7. **The Alberta plan was structurally vindicated, mechanistically superseded.** The prediction decomposition was correct in 2011; the missing mechanisms — joint representation learning, multi-step imagination, attention — took until 2023 to develop. "Ahead of its time" means structurally correct before the tools existed. It does not mean prescient in every detail.

---

## References

[1] Sutton, R. S., Modayil, J., Delp, M., Degris, T., Pilarski, P. M., White, A., & Precup, D. (2011). **Horde: A Scalable Real-time Architecture for Learning Knowledge from Unsupervised Sensorimotor Interaction.** *Proceedings of the International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS 2011).* University of Alberta / McGill University.

[2] Schlegel, M., Jacobsen, A., Abbas, Z., Patterson, A., White, A., & White, M. (2021). **General Value Function Networks.** *Journal of Artificial Intelligence Research*, 70, 497–543. University of Alberta / Amii.

[3] Chandrasekar, S., & Machado, M. C. (2025). **Towards An Option Basis To Optimize All Rewards.** *Reinforcement Learning Journal*, 2025. University of Alberta / Amii.

[4] Kotamreddy, H., & Machado, M. C. (2025). **A Study of Value-Aware Eigenoptions.** *Workshop on Inductive Biases in Reinforcement Learning, Reinforcement Learning Conference (RLC) 2025.* arXiv:2507.09127. University of Alberta / Amii.

[5] Pan, M., Zhang, W., Chen, G., Zhu, X., Gao, S., Wang, Y., & Yang, X. (2025). **Continual Visual Reinforcement Learning with A Life-Long World Model.** arXiv:2303.06572v2. Shanghai Jiao Tong University.

[6] Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). **Mastering Diverse Domains through World Models.** *Google DeepMind, University of Toronto.*

[7] Micheli, V., Alonso, E., & Fleuret, F. (2023). **Transformers Are Sample-Efficient World Models.** *International Conference on Learning Representations (ICLR 2023).* University of Geneva. arXiv:2209.00588.

[8] Hansen, N., Wang, X., & Su, H. (2022). **Temporal Difference Learning for Model Predictive Control.** *Proceedings of the 39th International Conference on Machine Learning (ICML 2022),* PMLR 162. UC San Diego.

[9] Khetarpal, K., Riemer, M., Rish, I., & Precup, D. (2022). **Towards Continual Reinforcement Learning: A Review and Perspectives.** *Journal of Artificial Intelligence Research*, 75, 1401–1476. Mila / McGill University / IBM Research / DeepMind. arXiv:2012.13490.

[10] Klissarov, M., Bagaria, A., Luo, Z., Konidaris, G., Precup, D., & Machado, M. C. (2025). **Discovering Temporal Structure: An Overview of Hierarchical Reinforcement Learning.** arXiv:2506.14045. Mila / McGill University / Amazon / Brown University / University of Alberta.
