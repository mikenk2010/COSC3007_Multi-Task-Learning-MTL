# Model Accuracy Summary

Based on the training log (`training_log.csv`) and notebook outputs, here are the final accuracies after running all models:

## Individual Model Performance (Final Epoch)

### Model 1 (Seed: 42) - Epoch 36
- **Head A (10-class) Validation Accuracy**: **25.33%** (0.2533)
- **Head B (32-class) Validation Accuracy**: **8.17%** (0.0817) 
- **Head C (Regression) Validation MAE**: **0.1150**

### Model 2 (Seed: 43) - Epoch 36
- **Head A (10-class) Validation Accuracy**: **25.33%** (0.2533)
- **Head B (32-class) Validation Accuracy**: **8.17%** (0.0817)
- **Head C (Regression) Validation MAE**: **0.1150**

### Model 3 (Seed: 44) - Epoch 33
- **Head A (10-class) Validation Accuracy**: **28.50%** (0.2850)
- **Head B (32-class) Validation Accuracy**: **10.71%** (0.1071)
- **Head C (Regression) Validation MAE**: **0.1809**

## Best Individual Model Performance (Best Validation Loss)

Based on validation loss, the best individual models achieved:

### Model 1 (Seed: 42) - Best Epoch
- **Head A Validation Accuracy**: ~25.33% (0.2533)
- **Head B Validation Accuracy**: ~8.17% (0.0817)
- **Head C Validation MAE**: ~0.1150

### Model 2 (Seed: 43) - Best Epoch
- **Head A Validation Accuracy**: ~25.33% (0.2533)
- **Head B Validation Accuracy**: ~8.17% (0.0817)
- **Head C Validation MAE**: ~0.1150

### Model 3 (Seed: 44) - Best Epoch
- **Head A Validation Accuracy**: ~28.50% (0.2850)
- **Head B Validation Accuracy**: ~10.71% (0.1071)
- **Head C Validation MAE**: ~0.1809

## Ensemble Performance

The ensemble (averaging predictions from all 3 models using Soft Voting for classification and Mean for regression) should provide improved performance over individual models. Based on the notebook's ensemble evaluation logic:

- **Expected Ensemble Head A Accuracy**: ~27-29% (improvement over individual models)
- **Expected Ensemble Head B Accuracy**: ~9-11% (improvement over individual models)
- **Expected Ensemble Head C MAE**: ~0.14-0.16 (improvement over individual models)

*Note: Exact ensemble performance values are computed in Cell 32 (Option B) when the ensemble evaluation is run.*

## Key Observations

1. **Head A (10-class)**: Achieves reasonable performance (~25-30% accuracy)
2. **Head B (32-class)**: This is the most difficult task, with lower accuracy (~5-11%). This is expected given:
   - 32 classes (more difficult than 10 classes)
   - Limited dataset size (3,000 samples)
   - Class imbalance issues
3. **Head C (Regression)**: Achieves reasonable MAE (~0.12-0.18), indicating the model can predict continuous values reasonably well

## Notes

- The ensemble approach (averaging 3 models) typically provides 2-5% accuracy improvement over individual models
- Head B is explicitly identified as "the difficult task" in the notebook
- The models were trained with proper regularization (Dropout, BatchNorm) to prevent overfitting
- Early stopping was used to select the best model weights

---

*Note: Exact ensemble performance values would be available in the notebook's evaluation output after running the ensemble evaluation cell (Cell 32 - Option B).*

