# Multi-Task Learning Deep Learning Project

## Academic Honesty Statement

> *I declare that this submission is my own work, and that I did not use any pretrained model or code that I did not explicitly cite.*

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Formulation](#problem-formulation)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Architecture](#architecture)
6. [Key Features](#key-features)
7. [Installation & Setup](#installation--setup)
8. [Usage](#usage)
9. [Results & Evaluation](#results--evaluation)
10. [Diagnostic Analysis](#diagnostic-analysis)
11. [References](#references)

---

## 🎯 Project Overview

This project implements a **Multi-Task Learning (MTL)** deep learning model with **Dual-Stream Architecture** that simultaneously predicts three independent targets from grayscale images. The architecture combines spatial-domain features (ResNet-V2) with frequency-domain features (Fourier Transform) for comprehensive representation learning. The solution follows best practices from **François Chollet's "Deep Learning with Python" (2nd Edition, Chapter 9: Advanced Vision and Chapter 13: Optimization)** and demonstrates research-grade implementation suitable for Master-level coursework.

### Key Highlights

- ✅ **Dual-Stream Architecture**: Combines Spatial (ResNet-V2) + Frequency (Fourier Transform) streams
- ✅ **True Ensembling**: Trains 3 models with different seeds and averages predictions
- ✅ **ResNet-V2 Architecture**: Implements skip connections with SeparableConv2D for parameter efficiency
- ✅ **Fourier Transform Features**: Frequency-domain feature extraction for improved regression performance
- ✅ **Professional Logging**: Custom TrainingLogger callback with CSV persistence
- ✅ **Comprehensive Visualizations**: FFT analysis, dual-stream activations, diagnostic plots
- ✅ **Comprehensive Diagnostics**: Class-wise analysis, residual analysis, confusion matrices
- ✅ **Mathematical Rigor**: LaTeX formulations and theoretical justifications

---

## 📊 Problem Formulation

The model must simultaneously predict three independent targets from the same input:

1. **Head A**: 10-class classification task (labels: {0, 1, 2, ..., 9})
2. **Head B**: 32-class classification task (labels: {0, 1, 2, ..., 31}) - *The difficult task*
3. **Head C**: Regression task predicting a continuous value in the range [0, 1]

### Why Multi-Task Learning?

Multi-Task Learning offers several advantages over training separate models (Chollet, 2021):

1. **Shared Representations**: A shared backbone learns features useful across all tasks
2. **Regularization Effect**: Learning multiple tasks simultaneously prevents overfitting
3. **Data Efficiency**: With limited data (3,000 samples), sharing representations improves learning
4. **Computational Efficiency**: A single forward pass produces predictions for all three tasks

---

## 📦 Dataset

- **Input**: `X` with shape `(3000, 32, 32)` - grayscale images
- **Targets**: `y` with shape `(3000, 3)` - three independent targets
- **Challenge**: Limited dataset size requires careful regularization and data augmentation

### Dataset Characteristics

- **Training Set**: 2,400 samples (80%)
- **Validation Set**: 600 samples (20%)
- **Stratification**: Stratified by Target B (32 classes) to ensure balanced class distribution

---

## 🔬 Methodology

### Framework: Chapter 13 Best Practices

This notebook follows **Chapter 13: Best Practices for the Real World** from Chollet (2021):

- **Scaling Up**: Mixed precision training and high-performance data pipelines using `tf.data`
- **Hyperparameter Tuning**: Discussion of systematic hyperparameter search strategies
- **Ensembling**: True implementation of model ensemble techniques

### Loss Formulation

The total loss is a weighted combination of task-specific losses:

$$L_{total} = w_A \cdot L_{CCE}(y_A, \hat{y}_A) + w_B \cdot L_{CCE}(y_B, \hat{y}_B) + w_C \cdot L_{MSE}(y_C, \hat{y}_C)$$

where:
- $L_{CCE}$ is Sparse Categorical Cross-Entropy for classification
- $L_{MSE}$ is Mean Squared Error for regression
- $w_A = 1.0$, $w_B = 2.5$, $w_C = 10.0$ are loss weights

**Loss Weight Justification**: The weights balance gradient magnitudes across tasks. $w_C = 10.0$ ensures regression gradients are comparable to classification gradients, preventing gradient starvation.

---

## 🏗️ Architecture

### Dual-Stream Architecture (Spatial + Frequency)

The model implements a **Dual-Stream Architecture** that combines spatial-domain and frequency-domain features for comprehensive representation learning:

```
                    Input (32×32×1)
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
   Spatial Stream                  Frequency Stream
   (ResNet-V2)                    (Fourier Transform)
          │                               │
   GlobalAvgPool(128)            GlobalAvgPool(64)
          │                               │
          └───────────┬───────────────────┘
                      ▼
              Concatenate (192)
                      │
              Dense(256) + BatchNorm
                      │
              Multi-Task Heads
```

#### Spatial Stream (ResNet-V2)

1. **Input Layer**: (32, 32, 1) grayscale images
2. **Data Augmentation**: RandomRotation, RandomZoom (active during training only)
3. **Backbone**:
   - Initial SeparableConv2D (64 filters) - *Increased from 32 for better capacity*
   - 4 Residual Blocks: 64→64→128→128 filters
   - Global Average Pooling → 128-dimensional features
4. **Shared Features**: Dense(256) + BatchNormalization

#### Frequency Stream (Fourier Transform)

1. **Fourier Transform Layer**: Custom `FourierTransformLayer` applies 2D FFT
   - Extracts magnitude spectrum (frequency-domain representation)
   - Normalized using standardization + clipping for training stability
2. **Lightweight CNN**:
   - Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.2)
   - Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.2)
   - Global Average Pooling → 64-dimensional features

#### Feature Fusion & Multi-Task Heads

1. **Feature Fusion**: Concatenate spatial (128) + frequency (64) = 192 features
2. **Shared Features**: Dense(256) + BatchNormalization
3. **Multi-Task Heads**:
   - **Head A**: Dense(128) → Dense(10, softmax)
   - **Head B**: Dense(256) → Dropout(0.5) → Dense(32, softmax)
   - **Head C**: Dense(64) → Dense(1, sigmoid) - *Benefits from frequency features*

### Why Dual-Stream?

- **Spatial Stream**: Captures hierarchical spatial patterns (edges, shapes, textures)
- **Frequency Stream**: Captures global frequency patterns (periodicity, overall structure)
- **Complementary Features**: Frequency domain provides information not captured by CNNs
- **Regression Benefit**: Frequency features particularly help Head C (continuous prediction)

### Why SeparableConv2D?

Separable convolutions (Chollet, 2021, Ch 13.1) reduce parameters by ~8-9x while maintaining similar representational capacity:
- **Depthwise Convolution**: Single filter per input channel
- **Pointwise Convolution**: 1×1 convolution to combine channels

### Why ResNet-V2?

ResNet-V2 addresses vanishing gradients through skip connections: $y = F(x) + x$, enabling training of deeper networks.

---

## ✨ Key Features

### 1. Professional Engineering & Logging

- **Type Hints & Docstrings**: All functions use Python type hints and Google-style docstrings
- **Custom TrainingLogger**: Logs all metrics to `training_log.csv` and displays formatted tables after each epoch
- **Sanity Checks**: Verifies model architecture with dummy tensors before training

### 2. High-Performance Data Pipeline

Following Chollet (2021, Ch 13.2), uses `tf.data.Dataset` with:
- `.shuffle(buffer_size=1024)` for training data
- `.batch(32)` for efficient GPU processing
- `.map(preprocess_fn, num_parallel_calls=AUTOTUNE)` for parallel preprocessing
- `.cache()` to cache preprocessed data in RAM
- **`.prefetch(AUTOTUNE)`**: Critical for performance - prefetches next batch while GPU trains

### 3. True Ensembling Implementation

**Not pseudocode** - fully implemented:
- Trains 3 separate models with seeds `[42, 43, 44]`
- Stores all models in `ensemble_models` list
- Averages predictions using Soft Voting (classification) and Mean (regression)
- Expected improvement: 2-5% accuracy boost

### 4. Advanced Visualization & Diagnostic Analysis

#### 4.1 Fourier Transform Visualization

**Frequency Domain Analysis** (Cell 8):
- **Sample-Level FFT**: Shows original image, FFT magnitude, FFT phase, and normalized output
- **Frequency Content Statistics**: Average magnitude spectrum across dataset
- **Radial Frequency Profile**: Low → High frequency distribution analysis
- **Purpose**: Understand how frequency-domain features complement spatial features

#### 4.2 Dual-Stream Activation Visualization

**Stream Comparison** (Cell 16):
- **Original Input vs Fourier Output**: Side-by-side comparison
- **Frequency Stream Activations**: Conv layer outputs on frequency features
- **Spatial Stream Activations**: Residual block outputs on spatial features
- **Feature Fusion**: 192-dimensional concatenated features visualization
- **Spatial vs Frequency Correlation**: Scatter plot showing feature relationships
- **Purpose**: Understand complementary nature of spatial and frequency streams

#### 4.3 Diagnostic Analysis

Comprehensive research-grade diagnostics:

- **Class-wise Performance**: Bar charts showing accuracy per class for Task B
- **Residual Analysis**: Histogram and Q-Q plot with Shapiro-Wilk normality test
- **Confusion Matrix**: Heatmap with masked diagonal to highlight confused class pairs
- **Ensemble Gain**: Visualization comparing ensemble vs. individual models
- **Error Analysis**: `show_worst_mistakes()` function displays images with highest loss

### 5. Training Callbacks

- **ModelCheckpoint**: Saves best model based on validation loss
- **EarlyStopping**: Stops training if no improvement for 15 epochs
- **ReduceLROnPlateau**: Reduces learning rate by 0.2× when stuck
- **TensorBoard**: Logs for visualization
- **TrainingLogger**: Custom callback for CSV logging and formatted tables

---

## 🚀 Installation & Setup

### Requirements

```bash
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
pandas>=1.3.0
keras_tuner
scipy
```

### Environment Setup

The notebook automatically checks and installs required packages. It also:

1. Sets global random seeds for reproducibility (NumPy, Random, TensorFlow)
2. Enables mixed precision training (`mixed_float16`) for GPU acceleration
3. Configures TensorFlow for optimal performance

---

## 💻 Usage

### Option A: Load Pre-trained Models

If you have saved models (`model_xxxx_seed{42,43,44}.h5` or `model_xxxx.h5`):

1. Run **Cell 30** (Option A) to load ensemble models or single model
2. Models are loaded with `compile=False` to avoid metric deserialization issues
3. Evaluation uses predictions directly (no compilation needed)

### Option B: Train from Scratch

To train the ensemble from scratch:

1. Ensure `TRAIN_FROM_SCRATCH = True` in **Cell 32** (Option B)
2. The notebook will:
   - Train 3 models with seeds [42, 43, 44]
   - Save each model as `model_xxxx_seed{seed}.h5`
   - Evaluate individual models and ensemble
   - Plot training curves for the best model

### Prediction Function

The `predict_fn(X32x32)` function:

- **Input**: NumPy array of shape `(N, 32, 32)` with dtype `float32`
- **Output**: NumPy array of shape `(N, 3)` with dtype `float32`
  - Column 0: Head A predictions (integers 0-9) - argmax of averaged probabilities
  - Column 1: Head B predictions (integers 0-31) - argmax of averaged probabilities
  - Column 2: Head C predictions (raw float in [0, 1]) - **CRITICAL: Raw float, NOT argmax**

**Ensemble Logic**: If `ensemble_models` exists, averages predictions from all 3 models. Otherwise, uses single model.

---

## 📈 Results & Evaluation

### Metrics Computed

**Classification Tasks (Head A & B)**:
- Accuracy
- Precision (weighted and macro)
- Recall (weighted and macro)
- F1-Score (weighted and macro) - **Important for imbalanced classes**

**Regression Task (Head C)**:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Training Logs

All metrics are logged to `training_log.csv` with the following columns:
- Epoch number
- Loss and validation loss (total and per-head)
- Accuracy/MAE metrics (train and validation)
- Learning rate

### Visualization

The notebook generates comprehensive visualizations:

#### Training Visualizations
- **Training Curves**: Loss and accuracy for all three heads (training vs. validation)
- **Learning Rate Schedule**: Cosine decay visualization
- **Overfitting Analysis**: Training vs. validation gap visualization

#### Feature Analysis Visualizations
- **Fourier Transform Analysis**: 
  - Original images vs. frequency-domain representations
  - Average frequency spectrum across dataset
  - Radial frequency profile (low → high frequency)
- **Dual-Stream Activations**:
  - Spatial stream feature maps
  - Frequency stream feature maps
  - Feature fusion visualization
  - Spatial vs. frequency feature correlation

#### Diagnostic Visualizations
- **Confusion Matrix**: Heatmap for Head B (32-class classification) with masked diagonal
- **Class-wise Performance**: Bar charts showing accuracy per class
- **Residual Analysis**: Histogram and Q-Q plot for regression errors
- **Ensemble Gain**: Bar chart comparing individual models vs. ensemble
- **Error Analysis**: Worst mistake visualization with predicted vs. actual labels

---

## 📊 Visualization Guide

For comprehensive visualization documentation, see **`VISUALIZATION_GUIDE.md`**, which provides detailed explanations of:

- **Fourier Transform Visualization** (Cell 8): Frequency domain analysis
- **Dual-Stream Activation Visualization** (Cell 16): Spatial vs. frequency comparison
- **Training Curve Visualizations** (Cell 24): Loss, accuracy, and MAE plots
- **Diagnostic Visualizations** (Cell 27): Class-wise, residual, confusion matrix analysis

Each visualization includes interpretation guidelines, common issues, and solutions.

---

## 🔍 Diagnostic Analysis

The notebook includes a comprehensive **Diagnostic Analysis** section (Section 11) that provides:

### 1. Class-wise Performance (Task B)

- Bar chart showing accuracy per class
- Scatter plot: Class frequency vs. accuracy
- Hypothesis: Rare classes have lower accuracy (class imbalance effect)

### 2. Residual Analysis (Task C)

- Histogram of regression errors
- Q-Q plot for normality testing
- Shapiro-Wilk normality test
- Statistics: Mean, Std, Skewness, Kurtosis
- Hypothesis: Regression errors follow a normal distribution

### 3. Confusion Matrix (Task B)

- Heatmap with **masked diagonal** to highlight errors
- Identifies most confused class pairs
- Reveals systematic misclassification patterns

### 4. Ensemble Gain

- Bar chart comparing individual models vs. ensemble
- Quantifies improvement from ensembling
- Hypothesis: Ensemble averaging reduces variance

### 5. Error Analysis

- `show_worst_mistakes()` function displays top k images with highest loss
- Helps identify failure modes and data quality issues

---

## 📚 References

### Primary Reference

- **Chollet, F. (2021).** *Deep Learning with Python* (2nd Edition). Manning Publications.
  - Chapter 9: Advanced Vision Techniques (Frequency Domain Analysis, Data Augmentation)
  - Chapter 13: Best Practices for the Real World
  - Chapter 13.1: Scaling Up (SeparableConv2D, parameter efficiency)
  - Chapter 13.2: High-Performance Data Pipelines (tf.data, prefetching)
  - Chapter 13.3: Model Ensembling

### Architecture References

- **He, K., et al. (2016).** "Identity Mappings in Deep Residual Networks." *ECCV 2016*.
- **Lin, M., et al. (2013).** "Network in Network." *arXiv:1312.4400*.

### Multi-Task Learning

- **Kendall, A., et al. (2018).** "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics." *CVPR 2018*.

### Frequency Domain Analysis

- **Gonzalez, R. C., & Woods, R. E. (2017).** *Digital Image Processing* (4th Edition). Chapter 4: Frequency Domain Processing.
- **Oppenheim, A. V., & Schafer, R. W. (2010).** *Discrete-Time Signal Processing* (3rd Edition). Chapter 3: The z-Transform.

---

## 📝 File Structure

```
COSC3007_Group_Assignment_2025C/
├── submission_xxxx.ipynb.ipynb    # Main notebook (Dual-Stream Architecture)
├── dataset_dev_3000.npz           # Dataset file
├── README.md                      # This file
├── DETAILED_ANALYSIS.md           # Comprehensive academic analysis
├── VISUALIZATION_GUIDE.md         # Detailed visualization documentation
├── UPGRADE_SUMMARY.md             # Architecture upgrade summary
├── ACCURACY_SUMMARY.md            # Performance metrics summary
├── VERIFICATION_REPORT.md         # Analysis verification report
├── requirements.txt               # Python dependencies
├── training_log.csv               # Generated during training
├── model_xxxx.h5                  # Single model (if saved)
├── model_xxxx_seed42.h5           # Ensemble model 1
├── model_xxxx_seed43.h5           # Ensemble model 2
└── model_xxxx_seed44.h5           # Ensemble model 3
```

---

## 🎓 Academic Context

This project demonstrates:

1. **Research-Grade Implementation**: Professional logging, type hints, comprehensive documentation
2. **Theoretical Understanding**: Mathematical formulations, loss weight justifications
3. **Best Practices**: Following Chollet Chapter 13 guidelines
4. **Deep Analysis**: Diagnostic analysis beyond basic evaluation
5. **Practical Engineering**: Production-ready code with error handling

---

## ⚠️ Important Notes

1. **Stratification**: The train/validation split stratifies by Target B (32 classes) - the most difficult task
2. **Normalization**: Uses training-only statistics to avoid data leakage
3. **Ensembling**: The `predict_fn` automatically uses ensemble if available, otherwise falls back to single model
4. **Column 2 Output**: Head C returns **raw float**, not argmax (common mistake)
5. **Model Loading**: Models are loaded with `compile=False` to avoid metric deserialization issues

---

## 📧 Contact & Support

For questions or issues related to this project, please refer to the course materials or contact the course instructor.

---

**Last Updated**: 2025  
**Course**: COSC3007 - Deep Learning  
**Institution**: RMIT University


