# Support Vector Machines (SVM) – Task 7

This project implements SVM classification (Linear and RBF kernels) using Scikit-learn.  
It includes preprocessing, hyperparameter tuning (C & gamma), cross-validation, evaluation metrics, ROC curves, and decision boundary visualization.

------------------------------------------------------------

## Project Structure

svm_task7.py                     # Main SVM script
svm_outputs/                     # Auto-generated output folder
│
├── confusion_matrix_linear.png
├── confusion_matrix_rbf.png
├── roc_linear.png
├── roc_rbf.png
├── svm_decision_boundary_rbf.png
├── svm_evaluation_summary.csv
├── svm_linear_pipeline.joblib
└── svm_rbf_grid_pipeline.joblib

------------------------------------------------------------

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn joblib

------------------------------------------------------------

## Features Implemented

1. Load & Preprocess Dataset
- Supports CSV and Excel files
- Auto-detects binary target column
- Drops empty or unnamed columns
- Handles missing values
- One-hot encodes categorical features
- Standardizes numeric features

2. Train SVM Models
- Linear SVM (LinearSVC / SVC linear)
- RBF kernel SVM

3. Hyperparameter Tuning (GridSearchCV)
- Tunes C and gamma
- Uses 5-fold Stratified cross-validation
- Selects best model using F1-score

4. Model Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix (saved as PNG)
- ROC curve + AUC score

5. Decision Boundary Visualization
- Plots decision boundary for 2D datasets
- For high-dimensional datasets: PCA (2 components)

6. Save Outputs
- All models, metrics, and plots saved to:

svm_outputs/

------------------------------------------------------------

## How to Run

1. Place your dataset inside the project folder  
   OR update the path inside the script:

RAW_PATH = "path/to/data.csv"

2. Run the script:

python svm_task7.py

3. All outputs will be generated inside:

svm_outputs/

------------------------------------------------------------

## Output Files Explained

- confusion_matrix_linear.png  
  Confusion matrix for Linear SVM.

- confusion_matrix_rbf.png  
  Confusion matrix for RBF SVM.

- roc_linear.png  
  ROC curve for Linear SVM.

- roc_rbf.png  
  ROC curve for RBF SVM.

- svm_decision_boundary_rbf.png  
  Decision boundary visualization (2D or PCA-reduced).

- svm_evaluation_summary.csv  
  Accuracy, precision, recall, F1-score, AUC for both models.

- svm_linear_pipeline.joblib  
  Saved Linear SVM pipeline.

- svm_rbf_grid_pipeline.joblib  
  Saved best RBF SVM model from GridSearchCV.

------------------------------------------------------------

## What You Learn

- SVM fundamentals  
- Linear vs RBF kernel differences  
- Margin maximization  
- Kernel trick  
- Hyperparameter tuning (C, gamma)  
- PCA-based visualization  
- Cross-validation evaluation  

------------------------------------------------------------

## Troubleshooting

Warning: "Skipping features without observed values"  
Cause: dataset has empty column (e.g., Unnamed: 32).  
Fix:

df = df.dropna(axis=1, how='all')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

Categorical encoding error  
Ensure categorical features are strings before encoding.

Missing ROC or Decision Boundary Plot  
If dataset cannot be reduced to 2D or has only 1 feature, SVM still trains but boundary cannot be plotted.

------------------------------------------------------------

## Next Steps & Enhancements

- Polynomial kernel SVM  
- Soft-margin visualizations  
- RandomizedSearchCV for faster tuning  
- Class imbalance handling (SMOTE, class weights)  
- Full Jupyter Notebook version with step-by-step explanation  

------------------------------------------------------------
