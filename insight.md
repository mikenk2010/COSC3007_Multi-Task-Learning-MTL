# Multi-Task Learning (MTL) – Experimental Insights

## Overview
This document summarizes findings from iterative experiments on a 3-task CNN model trained on 32×32 grayscale inputs.

Tasks:
- **Task A**: 10-class classification (global shape / identity)
- **Task B**: 32-class classification (orientation / fine-grained structure)
- **Task C**: Regression (intensity / amplitude)

The goal is to understand *learnability*, *task interference*, and *data limitations* before further architectural changes.

---

## Shared Backbone
Architecture:
- Conv(32) → MaxPool
- Conv(64) → MaxPool
- Conv(128)

### Observations
- Backbone consistently learns meaningful representations.
- Task A improves steadily across runs, indicating shared features are not degenerate.
- No evidence of backbone collapse or gradient domination from any single task.

Conclusion:
> The shared feature extractor is **healthy and sufficient** for at least one classification task.

---

## Task A – Global Shape (10 classes)

### Empirical Results
- Train accuracy: ~23–28%
- Validation accuracy: closely tracks training
- Loss decreases smoothly
- No significant overfitting

### Interpretation
- Task A is **well-posed** and visually learnable.
- GlobalAveragePooling works well for this task.
- Task A acts as a reliable signal that the model is learning.

Conclusion:
> Task A is **not the bottleneck** and should remain architecturally simple.

---

## Task C – Intensity / Amplitude (Regression)

### Empirical Results
- MAE ~0.22–0.24 (train & validation)
- Stable across epochs
- Insensitive to most architectural changes

### Techniques Used
- `tf.stop_gradient(x)` to prevent Task C from influencing shared features
- Lightweight head (GAP → Dense)

### Interpretation
- Task C is either:
  - easy, or
  - weakly dependent on spatial structure
- Regression signal is stable but low-information

Conclusion:
> Task C should remain **isolated and lightly weighted** to avoid gradient interference.

---

## Task B – Orientation / Fine Structure (32 classes)

### Empirical Results
- Train accuracy: ~4–5%
- Validation accuracy: ~2.5–3.5%
- Near random baseline (1 / 32 ≈ 3.1%)
- Loss plateaus early (~3.46)

### Key Findings

#### 1. Not Overfitting
- Train and validation accuracies are both low.
- Indicates **under-learning**, not memorization.

#### 2. Flatten + Dense is Insufficient
- Flattening spatial maps destroys relative spatial relationships.
- Orientation and fine-grained tasks are spatially sensitive.

#### 3. Oracle Dependency (Earlier Experiments)
- When Task B was conditioned on Task A labels:
  - Accuracy increased artificially
  - Model learned shortcuts (label leakage)
- Removing or weakening oracle input revealed true difficulty.

#### 4. Visual Signal May Be Weak
- Task B may be:
  - visually ambiguous
  - dependent on higher resolution
  - dependent on Task A context
  - noisy or inconsistently labeled

Conclusion:
> Task B is **fundamentally harder** and may be **data-limited**, not model-limited.

---

## Cross-Task Interaction

### Observations
- Task A learning does not help Task B automatically.
- Task C gradients can interfere unless explicitly stopped.
- Increasing Task B loss weight does **not** fix under-learning.

Conclusion:
> Multi-task setup is stable, but **positive transfer to Task B does not happen naturally**.

---

## What We Know for Sure

- The code is correct and stable.
- Training dynamics are sane.
- Task A and C are learnable.
- Task B is the true bottleneck.
- Task B failure is **not** caused by:
  - optimizer
  - loss function
  - learning rate
  - overfitting
  - model wiring bugs

---

## Open Questions (Next Session)

1. Is Task B visually distinguishable at 32×32 resolution?
2. Does Task B accuracy improve when:
   - conditioned on *predicted* Task A?
   - evaluated only on correctly predicted Task A samples?
3. Are Task B labels noisy or imbalanced?
4. Would Task B benefit from:
   - higher input resolution?
   - contrastive or metric learning?
   - grouping / hierarchical labels?

---

## Next Experiments (Deferred)

- Confusion matrix for Task B
- Accuracy of B conditioned on correct A
- Entropy analysis of B predictions
- Dataset visualization per B class
- Optional: freeze backbone, train B-only head

---

**Status**: Architecture exploration paused  
**Decision**: Investigate Task B data & label structure before further model changes
