# Target Attribute Characterization

This study employs a **multi-task convolutional neural network (MTL-CNN)** to predict three distinct targets (A, B, and C) from a shared visual input. Although the three targets are learned jointly, they differ significantly in terms of **spatial dependency**, **feature scale**, and **task difficulty**, which has important implications for both model design and performance.

---

## Target A — Global Shape / Geometric Structure

### Attribute Description
Target A represents **global geometric characteristics** of the object or pattern within the image. These attributes are determined by the **overall spatial arrangement** rather than localized details.

Typical examples include:
- Overall shape or form
- Coarse spatial layout
- Structural symmetry or topology
- Dominant global features

### Feature Dependency
- Primarily depends on **low- to mid-frequency spatial information**
- Relatively robust to local noise and small perturbations
- Moderately invariant to orientation changes
- Effectively captured by **deep convolutional features with global pooling**

### Learning Behavior
- Consistently outperforms the random baseline
- Shows steady learning during early and mid training stages
- Begins to plateau or slightly overfit in later epochs
- Indicates moderate competition with other tasks in the shared backbone

### Architectural Implications
Target A benefits from:
- Shared convolutional representations
- Global Average Pooling (GAP)
- Moderately regularized dense layers

Fine-grained spatial resolution is not critical at later network stages.

---

## Target B — Orientation / Fine-Grained Spatial Structure

### Attribute Description
Target B captures **fine-scale structural and orientation-sensitive attributes**, which depend on **precise spatial relationships** within the image.

Typical examples include:
- Orientation or angular alignment
- Directional patterns
- Local edge configuration
- Subtle structural class differences

### Feature Dependency
- Strong reliance on **high-frequency spatial features**
- Requires preservation of relative spatial positioning
- Highly sensitive to pooling operations
- Limited inherent rotation invariance

### Learning Behavior
- Training accuracy exceeds random baseline, confirming learnability
- Validation accuracy remains close to baseline
- Validation loss increases over time, indicating overfitting
- Strong evidence of negative transfer from other tasks

### Architectural Implications
Target B requires:
- Reduced or delayed pooling
- Spatially aware feature representations
- Task-specific convolutional branches
- Careful gradient management to reduce interference

A purely GAP-based head is insufficient for this target.

---

## Target C — Intensity / Amplitude (Global Scalar Attribute)

### Attribute Description
Target C corresponds to a **global scalar property** of the image, such as intensity or amplitude, and is independent of spatial arrangement.

Typical examples include:
- Mean or total signal intensity
- Amplitude-related measures
- Global magnitude statistics

### Feature Dependency
- Depends mainly on **global statistical information**
- Minimal sensitivity to spatial transformations
- Robust to orientation, translation, and pooling
- Easily separable from structural features

### Learning Behavior
- Converges rapidly during early training
- Achieves low Mean Absolute Error (MAE)
- Stable training and validation performance
- Minimal interference with other tasks

### Architectural Implications
Target C benefits from:
- Early Global Average Pooling
- A shallow regression head
- Optional gradient isolation (`stop_gradient`) to avoid task interference

It is the **least complex task** among the three.

---

## Comparative Summary

| Aspect | Target A | Target B | Target C |
|------|---------|----------|----------|
| Attribute Type | Global geometry | Orientation / fine structure | Global scalar |
| Spatial Sensitivity | Medium | High | Low |
| Dominant Frequencies | Low–mid | High | Very low |
| Pooling Tolerance | High | Low | Very high |
| Learning Difficulty | Moderate | High | Low |
| Generalization | Stable but limited | Poor without specialization | Strong |
| Task Interference | Moderate | High | Minimal |

---

## Key Insight for Multi-Task Learning

The three targets occupy **distinct regions of the feature spectrum**:

- **Target A** relies on global structural abstraction  
- **Target B** requires localized, orientation-preserving representations  
- **Target C** depends on global statistical aggregation  

As a result, a fully shared representation is **sub-optimal** without task-specific architectural adaptations, particularly for **Target B**, which is the most sensitive to spatial feature degradation.
