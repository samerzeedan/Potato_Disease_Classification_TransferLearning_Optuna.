# Potato_Disease_Classification_TransferLearning_Optuna.
Potato Leaf Disease Classification using EfficientNet (B0–B7) + Optuna Optimization

This project builds a high-performance deep learning model for classifying potato leaf diseases using transfer learning with EfficientNet architectures (B0 to B7) and automatic hyperparameter optimization using Optuna.

The final system achieves research-level accuracy after merging multiple datasets, applying heavy augmentation, and fine-tuning the best model selected by Optuna.

1. Datasets Used

Two datasets were merged into one unified dataset with 3 classes:

Early Blight

Late Blight

Healthy

Datasets:

PLD_3_Classes_256

Potato Leaf Disease Dataset (PlantVillage Format)

After Merging:

Train: 4151 images

Validation: 716 images

Test: 705 images

All images were resized dynamically based on each EfficientNet architecture (e.g., 224, 240, … 600).

2. Data Augmentation

Strong augmentation was applied to improve generalization:

Rotation

Width/height shifting

Zoom

Shear

Horizontal flip

Brightness variation

Fill-mode interpolation

This helped the model become robust against lighting, angles, and leaf variations.

3. Model Architectures (B0–B7)

All EfficientNet variants from B0 to B7 were tested:

B0, B1, B2, B3, B4, B5, B6, B7


Each model was loaded with ImageNet weights and extended with:

Global Average Pooling

Dropout

Dense softmax classifier

Then fine-tuning was applied based on Optuna’s recommendation.

4. Hyperparameter Optimization (Optuna)

Optuna was used to automatically search for the best combination of:

Architecture (B0 → B7)

Optimizers:
adam, adamw, sgd, rmsprop, nadam

Learning rate

Dropout rate

Fine-tuning percentage

Batch size

Best Trial Result:
Architecture: EfficientNet-B4
Optimizer: RMSprop
Learning Rate: 3.0526e-04
Dropout: 0.2259
Fine-tuning: 46%
Batch Size: 8


EfficientNet-B4 produced the strongest balance of accuracy + speed + generalization.

5. Final Model Performance
Validation Set

Accuracy: 99.86%

Loss: Very low (≈ 0.004–0.009)

Test Set

Accuracy: ~99–100%

F1 Score: ≈ 1.00

Perfect confusion matrix across 3 classes

This confirms that the model generalizes extremely well on unseen data.

6. Files Included

The repository includes:

best_model.h5 – final trained model

best_weights.h5 – model weights only

optuna_best_params.json – Optuna best hyperparameters

training_history.csv – accuracy/loss per epoch

confusion_matrix.png

classification_report.txt

notebook.ipynb – full end-to-end pipeline

8. Summary

Multiple datasets merged into one high-quality dataset

EfficientNet architectures B0–B7 evaluated

Optuna used for automated hyperparameter tuning

Best model: EfficientNet-B4 + RMSprop

Achieved near-perfect performance on validation and test sets

Code, weights, results, and evaluation are fully reproducible
