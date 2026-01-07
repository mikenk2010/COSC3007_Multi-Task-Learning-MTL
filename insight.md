Model: latest multi-head CNN (A: 10-class, B: 32-class, C: regression)

Dataset & setup
Input: 32×32 grayscale
Shared backbone: shallow–mid depth CNN with pooling
Training stable, no divergence, learning rate 1e-3
Loss-weighted multi-task objective

Target A (primary classification)
Best train accuracy ≈ 33–34%
Best validation accuracy ≈ 31–32%
Train/val gap small → good generalization, low overfitting
Performance plateau suggests backbone capacity, not optimization, is current limiter
Auxiliary tasks help stabilize early learning but cap peak accuracy
Single-head A-only model likely to outperform this if fully tuned

Target B (fine-grained / orientation-like classification)
Very low accuracy early, slow improvement
Best train accuracy ≈ 10–11%
Validation ≈ 6–7%, consistently lagging
Task appears harder, noisier, or weakly aligned with shared features
Strong candidate for gradient interference with A
Needs either deeper branch, higher resolution features, or decoupling from backbone

Target C (regression / intensity)
Fast convergence, low MAE (~0.15)
Acts as a regularizer rather than a performance target
Stop-gradient effectively prevents it from dominating shared features
Helpful for training stability, not for boosting A ceiling

Cross-task interactions
C stabilizes representation learning
B competes with A and consumes capacity
A benefits from multi-task early but saturates earlier than expected
Overall behavior consistent with auxiliary-task regularization + gradient conflict

Key conclusions
Current model is well-regularized but capacity-limited for A
B is the weakest and most harmful task in current form
Best use of this model: feature pretraining

Next steps (when continuing)
Use this model as pretrained backbone, then fine-tune A-only
Ablate B entirely or gate its gradients
Increase backbone depth or width only after B is fixed or removed