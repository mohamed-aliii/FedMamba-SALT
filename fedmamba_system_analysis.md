# FedMamba-SALT: System Analysis & Deep Origin Pathology

This document synthesizes the complete historical, technical, and mathematical trajectory of the FedMamba-SALT system. It strips away the symptoms and isolates the true deep origin of the structural failures we have encountered.

---

## 1. System Objective
**Goal:** To federate and fine-tune a pre-trained Self-Supervised Learning (SSL) State Space Model (FedMamba-SALT) on a distributed COVID-19 medical dataset across 12 clients characterized by **Extreme Label Skew** (mono-class and missing-class clients).

**The Challenge:** Maintain the continuous-time ODE dynamics of the Mamba blocks globally, while allowing the model to acquire the spatial/channel adaptations required to map highly entangled SSL medical features into a linearly separable space for the COVID-19 domains.

---

## 2. The History of Attempted Fixes & Why They Failed

### Attempt 1: Federated Branch Protect + Standard FedAvg
* **What we did:** Froze the `mamba_` ODE parameters to protect them from FedAvg. Left the shallow CNN layers (3.87M params) and the linear classifier unfrozen. Aggregated the whole matrix via standard FedAvg.
* **Why it failed (Missing-Class Collapse):** Multi-class clients lacking Class 2 (Clients 7-11) possessed massive datasets. Due to Cross-Entropy and Label Smoothing, their local optimization actively pushed the missing Class 2 logit to $-\infty$. Because of their massive sample weighting in standard FedAvg, their $-\infty$ gradients completely overwhelmed the healthy clients, destroying the global Class 2 boundary (leading to only 14 Class 2 predictions globally).

### Attempt 2: The "Walkarounds" (LayerNorm & Row-Wise Aggregation)
* **What we did:** Replaced `BatchNorm1d` with `LayerNorm`, and implemented `class_head_only` proportional row-wise aggregation to "protect" the Class 2 row from the missing-class clients.
* **Why it failed (Decoupled Softmax Corruption):**
  1. **LayerNorm Destruction:** Normalizing across the 768-D feature vector independently per-sample destroyed the absolute magnitude of the SSL features, crippling the discriminator.
  2. **The Frankenstein Classifier:** By assembling the Class 2 Softmax numerator purely from mono-class clients (who pushed it to $+\infty$), while assembling the denominators (Class 0 and 1) from multi-class clients, we fundamentally decoupled the Softmax function. We destroyed the relative logit scaling, artificially inflating Class 2 and causing a massive explosion (3030 predictions).

### Attempt 3: Equal Class-Wise Weighting
* **What we did:** Proposed enforcing `--client_weighting equal` directly onto the row-wise aggregation to dilute the mono-class $+\infty$ bias.
* **Why it failed (Symptom Masking):** The user correctly rejected this. Equal weighting merely dilutes the Frankenstein decoupling; it is a mathematical band-aid that fails to address the root pathology of why the logits are exploding in the first place.

### Attempt 4: Architectural Tear-Downs (FedBABU, Cosine Classifiers, Prototypes)
* **What we did:** Proposed bounding the logits geometrically by using L2-Normalized Cosine Heads, or freezing the classifier entirely (FedBABU).
* **Why it failed (Generating New Problems):** 
  - **FedBABU:** Regresses to the failed "Linear Probe" experiment. Freezing a random global head forces mature SSL features to warp into arbitrary projections, destroying their pre-trained spatial geometry.
  - **Cosine Heads:** Scrambles the linearly-pre-trained SSL representation manifold. These fixes generated massive new architectural mismatches.

### Attempt 5: Restricted Softmax (Masked Cross-Entropy / FedRS)
* **What we did:** Proposed dynamically masking the logits of missing classes during local training to mathematically force their gradients to $0.0$, preventing the $-\infty$ push without hacking the aggregation logic.
* **Why it failed (Recycling Failed Experiments):** As the user noted, "we did that before and it failed and this is workaround." Masking logits stops the missing-class destruction, but it mathematically isolates the classes. It forces the mono-class clients to learn nothing in the classifier (gradient = 0), totally wasting their data and failing to align their features globally. 

---

## 3. The Deep Origin Source of the Problems

Through exhaustive process of elimination, we have isolated the bedrock failure. The classifier head exploding ($+\infty$) or collapsing ($-\infty$) is **not the root problem; it is a symptom of a critically bottlenecked encoder.**

### The Encoder Capacity vs. Federation Paradox
1. **The Representation Reality:** `diagnostic_teacher_probe.py` proved that the pre-trained SSL features are **not linearly separable**. To separate the COVID classes, the encoder MUST perform deep, non-linear transformations on the feature space.
2. **The Capacity Bottleneck:** The Mamba blocks contain the network's true global spatial and long-range sequence modeling capacity. The shallow CNN layers alone do not possess the capacity to disentangle the SSL manifold.
3. **The Mamba Federation Trap:** If we unfreeze the Mamba parameters, standard FedAvg linearly averages the discrete transition matrices ($\bar{A}, \bar{B}$). This mathematically destroys the continuous-time ODE dynamics, leading to representation collapse. 

### The Ultimate Conclusion
Because the Mamba blocks are frozen (Federated Branch Protect), the encoder lacks the capacity to map the images into a linearly separable space. Because the features remain entangled, the `nn.Linear` classifier simply cannot draw a valid boundary. 
**To artificially force the Cross-Entropy loss down on inseparable features, the local optimizer has no choice but to resort to pathological magnitude explosions ($+\infty$ / $-\infty$).**

### The True Objective Moving Forward
All classifier aggregation hacks, logit masking, and Softmax manipulations are ultimately futile walkarounds because they treat the symptom of an entangled feature space.

**We must find a way to impart learning capacity to the Mamba blocks in a federated setting WITHOUT subjecting their core continuous-time ODE parameters to destructive linear averaging.** 

*(e.g., Parameter-Efficient Fine-Tuning / LoRA injected strictly into the Mamba block's linear projections, allowing the block to adapt non-linearly while leaving the fragile ODE state-transition matrices perfectly frozen.)*
