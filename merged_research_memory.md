Historical Research Context (Pre-FedNMC Era)
Project State
Objective

Improve federated medical image classification performance on Retina and COVID-FL while preserving convergence under heterogeneous client distributions.

Status

Migration from Retina-focused experimentation toward COVID-FL benchmark completed.

Multiple training pipelines audited and corrected.

True baselines established.

Main Blockers
Hidden dataset-specific hardcodes.
Incorrect CLI argument propagation.
Client metadata inconsistencies.
Instability during optimizer warm starts.
Severe non-IID class imbalance.
Critical Context

Performance regressions were frequently caused by implementation issues rather than algorithmic limitations.

Repeated audits showed many assumptions about dataset portability were false.

Important Decisions
Decision

Separate dataset logic from Retina-specific code.

Reason
COVID-FL migration exposed numerous hidden Retina assumptions.

Evidence
Training failures and incorrect splits occurred when swapping datasets.

Status
Completed.

Decision

Audit notebooks before proposing algorithmic improvements.

Reason
Observed performance anomalies were often implementation bugs.

Evidence
Multiple hardcoded paths and dataset assumptions discovered.

Status
Completed.

Decision

Establish verified baselines before introducing new FL methods.

Reason
Research comparisons were invalid without trustworthy baselines.

Evidence
Several reported results changed after code corrections.

Status
Completed.

Decision

Investigate SCAFFOLD as a primary heterogeneity mitigation method.

Reason
Client drift appeared to be a dominant failure mode.

Evidence
Theoretical suitability and empirical observations.

Status
Explored but not final solution.

Technical Knowledge
Dataset Migration Findings
Retina pipeline not directly portable to COVID-FL.
Dataset-specific augmentations influenced convergence significantly.
Client split generation required verification.
Metadata preservation critical for reproducibility.
Training Pipeline Findings
Hidden hardcoded dataset references existed.
CLI parser handling of --dataset contained issues.
Notebook defaults sometimes overrode user parameters.
Several experiments unintentionally used incorrect configurations.
Optimization Findings
Warm-start AdamW produced instability and NaN failures.
Learning-rate sensitivity increased under non-IID distributions.
SCAFFOLD partially reduced drift but did not fully solve performance limitations.
Evaluation Findings
TTA and augmentation choices materially affected measured accuracy.
Baseline verification changed interpretation of algorithmic gains.
Experiment Registry
Experiment

COVID-FL migration audit

Changes
Replace Retina assumptions.

Outcome
Multiple incompatibilities discovered.

Interpretation
Codebase required dataset-agnostic refactoring.

Experiment

Baseline revalidation

Changes
Re-run corrected pipelines.

Outcome
Verified:

Retina ≈ 81.93%
COVID-FL ≈ 91.47%

Interpretation
Several earlier comparisons were unreliable.

Experiment

Warm-start AdamW

Changes
Initialize from previous federated states.

Outcome
NaN divergence.

Interpretation
Unsafe under current setup.

Experiment

SCAFFOLD evaluation

Changes
Introduce control variates.

Outcome
Improved stability but limited overall gains.

Interpretation
Drift reduction alone insufficient.

Dead Ends
Warm-start optimizer strategy

Failed due to instability and NaNs.

Confidence: High.

Assumption that Retina pipeline generalizes directly

Invalid.

Confidence: High.

Assuming notebook defaults matched CLI parameters

Invalid.

Confidence: High.

Constraints
Technical
Non-IID client distributions.
Limited communication rounds.
Medical imaging domain.
Compute
Graduation-project-scale resources.
No large cluster availability.
Data
Fixed client partitions.
Class imbalance.
Research
Must outperform strong baselines.
Must remain implementable in existing framework.
Open Questions
Can drift mitigation alone solve performance loss?

Importance: High

Needed Evidence:
Controlled SCAFFOLD/FedProx comparisons.

Which failures are optimization vs representation failures?

Importance: High

Needed Evidence:
Encoder-level diagnostics.

How much improvement is available from calibration alone?

Importance: Medium

Needed Evidence:
FedLC-style studies.

Next Research Actions
Verify all training pipelines.
Lock trusted baselines.
Separate optimization failures from representation failures.
Evaluate drift-aware methods.
Investigate representation-preserving alternatives.
Compact Timeline

Retina experiments
→ COVID-FL migration
→ Notebook audits
→ Codebase audits
→ Baseline correction
→ Warm-start failures
→ SCAFFOLD exploration
→ Drift-focused analysis
→ Transition toward representation-centric solutions
→ Later FedLC / FedNMC research phase

Research Memory Snapshot

Retina→COVID-FL migration revealed substantial implementation debt (hardcodes, parser issues, dataset assumptions). Extensive auditing established trusted baselines (Retina ~81.93, COVID-FL ~91.47). Warm-start AdamW produced NaN divergence. SCAFFOLD reduced drift but did not fully address performance ceilings. Findings suggested that optimization/client-drift issues were only part of the problem, motivating later investigation into representation preservation, calibration methods, and eventually the FedNMC/FedLC research direction.

# FedMamba-SALT: Federated Medical Model Optimization — Research Memory

---

# 1. Project State

**Current objective:** Stabilize federated fine-tuning of a pre-trained SSL encoder (InceptionMamba) on COVID-FL dataset (12 clients, extreme non-IID / split_real) for 3-class classification (Normal, Pneumonia, COVID) without catastrophic forgetting of Class 2 (COVID).

**Current status:** Architecture pivoted from parametric nn.Linear classifier to non-parametric **FedNMC (Federated Nearest Mean Classifier)** with Cosine Softmax Contrastive Prototypes. 100-round run reached AUC ~0.80 but exhibits violent "Yo-Yo" oscillation in predictions. Optimizer flush (per-round fresh AdamW) deployed to address momentum explosion. Awaiting full-run verification.

**Main blockers:**
1. Prediction oscillation: pred array swings between Class 2 monopoly and vanishing (e.g., Round 10: `[2358, 1581, 35]` vs Round 11: `[736, 966, 2272]`)
2. Persistent AdamW momentum causing small clients (C12, 96 images) to act as "unguided missiles"
3. `val_loss=0.0000/0.0000` logged because prototype loss not wired into eval logging

**Critical context:**
- Encoder: InceptionMamba (SSM + CNN hybrid), embed_dim=96, feat_dim=768
- 17.90M total params, 0.1786M trainable (1%) via LoRA on proj_main/proj_gate/proj_out + 1x1 convs
- Head: `nn.Identity()` — 0 trainable head params
- Teacher: Frozen MAE ViT-B/16 (SALT loss not used in FedNMC phase)
- Client distribution is pathologically non-IID: C1/C6 mono-class Class 2 (2835 samples), C7-C11 have zero Class 2

---

# 2. Important Decisions

| # | Decision | Reason | Evidence | Status |
|---|----------|--------|----------|--------|
| 1 | Abandon parametric nn.Linear classifier | Softmax CE requires negative examples to draw boundaries; mono-class clients push missing-class logits to ±∞ | Class 2 collapsed to 0 predictions then exploded to 3030 | **Executed** |
| 2 | Adopt FedNMC with Cosine Softmax Prototypes | Prototype alignment needs zero negative examples; bounded by construction; Cosine Similarity prevents magnitude explosion | Diagnostic: AUC 0.71 → 0.85 in 24 rounds | **Executed** |
| 3 | Freeze all encoder normalizations (LayerNorm/GroupNorm) | Unfrozen affine params caused "covariate rotation" — mono-class clients twisted feature space each round | pred array Yo-Yo correlated with norm updates | **Executed** |
| 4 | Prototype EMA momentum (α=0.9) | Overwriting prototypes each round creates non-stationary loss surface | Oscillation observed in pred arrays | **Executed** |
| 5 | Relax temperature τ from 0.07 → 0.15 | τ=0.07 hypersensitive — tiny feature drift caused inf gradients (C12 loss hit 6.12) | Loss spikes in diagnostic logs | **Executed** |
| 6 | Flush AdamW optimizer per round (revert FIX-4) | Persistent momentum buffers caused stale gradients to fire on fresh global params | C12 update norms 10x larger than C9 despite 32x less data | **Executed** |
| 7 | Keep `--client_weighting equal` for encoder | Size weighting would give RSNA clients (74% of data, zero COVID) control of encoder, drowning out COVID features | Client counts: C7-C11 = 11,889 images, zero Class 2 | **Rejected size weighting** |
| 8 | Remove FedLC loss | tau * log(P) caused gradient starvation on mono-class clients (0.0000 loss on C1/C6) | c1=c6=0.0000 in PEFT-FedLC run | **Rejected / Reverted** |
| 9 | Row-wise class_head_only aggregation | Proportional sample weighting created "Frankenstein Classifier" — decoupled Softmax numerator/denominator | Class 2 over-predicted 3030 samples | **Rejected** |
| 10 | LoRA injection (rank=8) | Full fine-tuning destroys Mamba ODE state-transition matrices under FedAvg; freezing causes underfitting | Prior runs collapsed; 1% trainable via LoRA stabilizes | **Executed** |
| 11 | Freeze mamba_ parameters (branch protect) | Δ, A, B, C parameters define ODE via matrix exponentials; linear averaging destroys eigenvalue structure | Theoretical: FedAvg incompatible with SSM params | **Executed** |

---

# 3. Technical Knowledge

## Key Findings

1. **Class 2 Collapse Mechanism (Original):** CrossEntropyLoss gradient = p_i - y_i. For missing-class clients, y_i=0 → positive gradient always → optimizer subtracts from Class 2 weights → negative infinity. FedAvg averages these destroyed weights into global model.
2. **Class 2 Explosion Mechanism (Row-wise fix):** Mono-class clients (C1, C6) with no negative examples push Class 2 bias to +∞. Proportional weighting gave them 80.5% of Class 2 row → over-predicted 3030 samples.
3. **Frankenstein Classifier:** Row-wise aggregation decouples Softmax — assembles Class 2 numerator from mono-class clients but excludes their denominator updates → artificially inflates logits.
4. **Covariate Rotation:** Unfrozen LayerNorm/GroupNorm on mono-class clients twists the entire 768-D feature space locally. Global prototypes are static points in that space → rotation throws features out of alignment → predictions invert.
5. **Persistent Momentum Bomb:** AdamW buffers m_t accumulate gradients from old parameter space. After FedAvg broadcast, client applies stale m_t to new params → unguided parameter update. Small clients (1 batch) most affected.

## Architecture

- **Encoder:** InceptionMamba (two-branch: SSMBranch with Mamba SSM layers + InceptionBranch with 1x1/3x3/5x5 convolutions)
- **LoRA targets:** `proj_main`, `proj_gate`, `proj_out` (SSM linear layers) + 1x1 convolutions in InceptionBranch
- **Frozen:** Mamba SSM core params (A, B, C, Δ), all backbone norms
- **Head:** `nn.Identity()` — feature vectors used directly for Cosine Similarity against prototypes
- **Loss:** Cosine Softmax (temperature-scaled CrossEntropy over feature↔prototype dot products)
- **Aggregation:** `average_models()` for encoder LoRA params (trainable only, ~1-2M); `aggregate_prototypes_ema()` for class centroids

## Prototype Aggregation Math

```
C_c^(t) = α * C_c^(t-1) + (1-α) * C_new,c   where α=0.9
C_c^(t) ← normalize(C_c^(t), L2)             # re-project to unit sphere
C_new,c = Σ_k (N_kc / N_total,c) * μ_kc      # sample-weighted local centroids
```

## Inference
```
logits[b,c] = cosine_similarity(features[b], global_centroid[c])
pred = argmax(logits / τ)                    # τ=0.15
```

## Tradeoffs

| Approach | Pros | Cons |
|----------|------|------|
| FedNMC Prototypes | No negative examples needed; magnitude-bounded; naturally federated | Rigid single-centroid per class; oscillation risk; requires EMA |
| Parametric CE | Flexible decision boundary | Requires negative examples; explodes on mono-class clients |
| FedLC | Calibrates missing-class logits | Gradient starvation on mono-class; redundant with prototype approach |
| FedBABU (frozen head) | Prevents head corruption | Random init destroys SSL features; incompatible with pre-trained head |
| Full fine-tuning | Maximum capacity | Destroys Mamba ODE structure under FedAvg |

---

# 4. Experiment Registry

| Experiment | Configuration | Outcome | Interpretation |
|------------|--------------|---------|----------------|
| Baseline FedAvg + full finetune | Unfrozen encoder, standard CE, FedAvg | Class 2 collapses to 0 predictions after few rounds | Missing-class gradients destroy Class 2 weights via full-matrix averaging |
| class_head_only row-wise (proportional) | Row-wise aggregation by sample count | Class 2 explodes to 3030 predictions | Mono-class clients (80.5% of Class 2 data) inject +∞ bias |
| PEFT-FedLC | LoRA + FedLCLoss (tau=1.0) | C1/C6 loss=0.0000 (gradient starvation) | tau*log(P) → -∞ for P=1.0 on mono-class clients; no gradient signal |
| Frankenstein (Standard CE + L2 calibrate) | Server-side L2 norm equalization | Aborted — architecturally unsound | Decoupled row aggregation violates Softmax coupling |
| FedNMC v0 (no EMA, τ=0.07, unfrozen norms) | Prototypes + Cosine Softmax | AUC 0.71→0.85, violent Yo-Yo oscillation | Unfrozen norms rotate feature space; τ too sharp; momentum explosion |
| FedNMC v1 (EMA=0.9, τ=0.15, frozen norms, flushed opt) | All stabilization fixes | AUC 0.72→0.80, reduced oscillation, ongoing | Partial success — oscillation dampened but not eliminated |

---

# 5. Dead Ends

| Dead End | Why Failed | Confidence |
|----------|-----------|------------|
| BatchNorm1d in classifier | Running statistics diverge under non-IID; averaging corrupts global norm | High — replaced with LayerNorm then removed entirely |
| Row-wise proportional aggregation | Decouples Softmax numerator/denominator; creates Frankenstein classifier | High — mathematical proof in audit |
| FedLC (Logit Calibration) | Local prior P(y)=1.0 on mono-class → log(1)=0 → no gradient; total starvation | High — empirical c1=c6=0.0000 |
| FedBABU (frozen random head) | Forcing SSL features to match random projection destroys representations | High — theoretical |
| Pure MSE Prototype Pull (no contrastive) | Attractive-only force → catastrophic feature collapse to origin | High — mathematical proof |
| Global prior FedLC | Uses global P(y) instead of local; bypasses entire point of FedLC; adds dead complexity | High — rejected in audit |
| Excluding mono-class clients from head agg | Throws away 80.5% of Class 2 data; starves classifier | High — user explicitly rejected |
| Server-side L2 calibration on hyperplanes | Equalizes row norms but doesn't fix angular misalignment from blind training | High — workaround not fix |

---

# 6. Constraints

| Category | Constraint |
|----------|------------|
| Technical | mamba-ssm uses fused CUDA kernels — cannot LoRA into SSM core directly; must target surrounding linear layers |
| Technical | Feat_dim=768 fixed by encoder architecture; prototypes are 768-D unit vectors |
| Compute | Colab environment; full run ~260s/round for 12 clients |
| Data | 12 clients, pathologically non-IID: C1/C6 mono-class Class 2; C7-C11 zero Class 2; RSNA dominates by volume |
| Data | Global class distribution: Class 0: 7285, Class 1: 5237, Class 2: 3522 |
| Research | Must preserve FedMamba-SALT paper alignment; Mamba ODE theory is core contribution |
| Research | Cannot use centralized proxy data for server-side head training (violates FL premise) |

---

# 7. Open Questions

| Question | Importance | Needed Evidence |
|----------|-----------|-----------------|
| Does optimizer flush fully eliminate Yo-Yo oscillation? | Critical | Full 100-round run with flushed optimizers |
| Is single-prototype-per-class too rigid for COVID subtypes? | High | Compare against multi-prototype or Gaussian mixture approach |
| Does frozen norm hurt adaptation of CNN spatial filters? | Medium | Ablate: frozen vs unfrozen norms (with EMA) |
| Should prototype aggregation use equal vs proportional weighting? | Medium | Equal prevents minority erasure; proportional is statistically correct |
| Can we combine FedNMC feature alignment with a server-side linear probe at eval? | High | Run calibrated linear probe on frozen encoder features post-training |
| Is τ=0.15 optimal? | Medium | Grid search τ ∈ {0.07, 0.10, 0.15, 0.20, 0.30} |
| Does LoRA rank=8 provide enough capacity? | Medium | Ablation: r=4, 8, 16 |
| How to handle val_loss=0.0000 logging? | Low | Wire prototype loss into evaluate_global logging |

---

# 8. Next Research Actions

1. **Verify optimizer flush fix** — Run full 100-round diagnostic; monitor pred array stability and local loss smoothness *(highest priority)*
2. **Implement server-side calibrated linear probe** — After FedNMC training freezes, train nn.Linear on server validation features for 5-10 epochs; this draws optimal hyperplanes through the prototype-aligned space *(high priority — could unlock 90%+ AUC)*
3. **τ grid search** — Run τ ∈ {0.10, 0.15, 0.20} to find optimal sharpness/gradient-stability tradeoff *(medium priority)*
4. **LoRA rank ablation** — Run r=4, 16 to verify capacity vs stability tradeoff *(medium priority)*
5. **Wire prototype loss into eval logging** — Currently val_loss=0.0000; should log actual Cosine Softmax loss for monitoring *(low priority)*
6. **Multi-prototype exploration** — If single-prototype rigid, consider per-class Gaussian mixture or k-means centroids *(future work)*

---

# 9. Compact Timeline

| Round | Event | Key Metric / Decision |
|-------|-------|----------------------|
| 0 | Initial FedAvg baseline | Class 2 collapse identified |
| 1 | class_head_only phantom feature audit | Code verified: functionally identical to standard FedAvg |
| 2 | Row-wise proportional aggregation implemented | Class 2 explosion to 3030 predictions |
| 3 | Mono-class exclusion attempted | Rejected — loses 80.5% of COVID data |
| 4 | Deep problem identified: parametric CE incompatible with mono-class FL | Architecture pivot decision |
| 5 | FedNMC Prototype architecture designed | Cosine Softmax with L2-normalized features |
| 6 | PEFT-FedLC attempted (LoRA + FedLC) | Failed: gradient starvation on mono-class |
| 7 | Frankenstein (Standard CE + L2 calibrate) | Aborted: decoupled Softmax |
| 8 | FedNMC tear-down executed; 3-round diagnostic | Cold start success; AUC 0.71→0.85 |
| 9 | 50-round diagnostic | Yo-Yo oscillation identified; AUC plateau ~0.85 |
| 10 | Three stabilization fixes: frozen norms, EMA=0.9, τ=0.15 | Oscillation reduced |
| 11 | Optimizer flush fix deployed | Per-round fresh AdamW |
| 12 | 100-round run (diag5) | AUC 0.72→0.80, ongoing — awaiting full results |
| 13 | Server-side calibrated linear probe proposed | Next experiment |

---

# 10. Research Memory Snapshot (Ultra-Compact)

**Problem:** Extreme non-IID federated learning (mono-class clients) makes parametric classifiers mathematically impossible — Softmax CE requires negative examples.

**Solution:** FedNMC — replace nn.Linear with L2-normalized feature prototypes. Classification via Cosine Similarity. Local training uses Cosine Softmax (temp=0.15) against frozen global prototypes. Aggregation via EMA (α=0.9) on sample-weighted centroids with L2 re-projection.

**Architecture:** InceptionMamba encoder with LoRA (rank=8, 1% trainable) on projection layers + 1x1 convs. Mamba core frozen. All norms frozen. Head = Identity().

**Critical fixes applied:**
- Optimizer flush per round (no persistent AdamW momentum)
- Frozen encoder norms (prevent covariate rotation)
- Prototype EMA (prevent non-stationary targets)
- τ=0.15 (prevent hypersensitive gradient explosion)
- Equal client weighting for encoder (prevent COVID domain erasure)

**Current status:** AUC ~0.80, reduced oscillation. Next: verify flush fix + server-side linear probe eval.

**Files modified:** `train_fed_finetune.py`, `utils/fedavg.py`, `models/lora.py`, `docs/model_alignment.md`

**Key functions:**
- `aggregate_prototypes_ema()` — prototype aggregation with EMA + L2 renormalize
- `local_train_one_round()` — Cosine Softmax loss against frozen global_centroids
- `evaluate_global()` — Cosine Similarity inference against global centroids
- `inject_lora_into_encoder()` — LoRA wrapper injection

**The one-sentence insight:** In FL with mono-class clients, you cannot average hyperplanes drawn by the blind — instead, average the geometric centers of what they can see, and classify by angular proximity.
