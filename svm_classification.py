

import os
import fnmatch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------- CONFIG ----------------
RAW_PATH = "C:\\Users\\chand\\Downloads\\breast-cancer.csv"   # <-- uploaded dataset path
OUT_DIR = 'svm_outputs'
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Load dataset ----------------
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Dataset not found at {RAW_PATH}. Place file there or edit RAW_PATH.")

if RAW_PATH.lower().endswith(('.xls', '.xlsx')):
    df = pd.read_excel(RAW_PATH)
else:
    df = pd.read_csv(RAW_PATH)

print("Loaded dataset:", RAW_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ---------------- Detect binary target column ----------------
def detect_target_column(df):
    common_targets = ['target','label','y','class','diagnosis','outcome','status']
    for name in common_targets:
        for c in df.columns:
            if c.lower() == name:
                return c
    # any column with just 2 unique values
    for c in df.columns:
        if df[c].nunique(dropna=True) == 2:
            return c
    # last column fallback
    return df.columns[-1]

target_col = detect_target_column(df)
print("Detected target column:", target_col)
print(df[target_col].value_counts(dropna=False))

# ---------------- Prepare y (binary 0/1) ----------------
def binarize_series(s):
    if s.dtype.kind in 'biufc':
        uniq = np.unique(s[~pd.isna(s)])
        if set(uniq).issubset({0,1}):
            return s.astype(int)
        if len(uniq) == 2:
            # map smaller->0, larger->1
            mapping = {uniq[0]:0, uniq[1]:1}
            return s.map(mapping).astype(int)
    s_lower = s.astype(str).str.lower()
    pos = s_lower.isin(['1','yes','y','true','t','positive','malignant'])
    neg = s_lower.isin(['0','no','n','false','f','negative','benign'])
    out = pd.Series(np.nan, index=s.index)
    out[pos] = 1
    out[neg] = 0
    if out.notna().sum() >= len(s) * 0.5:
        # fill remaining by factorize if still two categories
        if out.isna().any():
            fac = pd.factorize(s)[0]
            uniq = np.unique(fac)
            if len(uniq) == 2:
                mapping = {uniq[0]:0, uniq[1]:1}
                return pd.Series(fac).map(mapping).astype(int)
        return out.astype(int)
    # fallback: factorize and map to 0/1 if binary
    fac = pd.factorize(s)[0]
    if len(np.unique(fac)) == 2:
        return pd.Series(fac).astype(int)
    raise ValueError(f"Could not binarize target column {s.name}")

y = binarize_series(df[target_col])
df = df.loc[y.notna()].reset_index(drop=True)
y = y.loc[y.notna()].reset_index(drop=True)

# Use 'target' as standardized name
df['target'] = y
X = df.drop(columns=[target_col, 'target']) if target_col in df.columns else df.drop(columns=['target'])

# remove id-like columns (all unique)
id_like = [c for c in X.columns if X[c].nunique() == X.shape[0]]
if id_like:
    print("Dropping id-like columns:", id_like)
    X = X.drop(columns=id_like)

# ---------------- Select feature types ----------------
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
print("Numeric columns:", numeric_cols)
print("Categorical columns:", cat_cols)

# For SVM we prefer numeric features. We'll one-hot encode categorical columns.
# If there are many categories, OneHot may explode; script will still attempt it.
# ---------------- Build preprocessing pipeline ----------------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# handle OneHotEncoder argument differences across sklearn versions
ohe_kwargs = {'handle_unknown': 'ignore'}
try:
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False, **ohe_kwargs)
except TypeError:
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse=False, **ohe_kwargs)

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', ohe)
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, cat_cols)
], remainder='drop')

# ---------------- Train/Test split ----------------
X_full = X.copy()
y_full = y.copy()
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_full
)
print("Train/Test shapes:", X_train.shape, X_test.shape)

# ---------------- SVM (Linear) pipeline ----------------
linear_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', LinearSVC(random_state=RANDOM_STATE, max_iter=5000))
])
print("\nTraining LinearSVC...")
linear_pipeline.fit(X_train, y_train)
y_pred_linear = linear_pipeline.predict(X_test)

# evaluate linear SVM
def print_eval(name, y_true, y_pred):
    print(f"\n=== {name} Evaluation ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1:", f1_score(y_true, y_pred, zero_division=0))
    print("Classification report:\n", classification_report(y_true, y_pred, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    return cm

cm_linear = print_eval("Linear SVM", y_test, y_pred_linear)

# save linear confusion matrix
plt.figure(figsize=(4,3))
sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Blues')
plt.title('Linear SVM Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix_linear.png'))
plt.close()
print("Saved confusion_matrix_linear.png")

# Save linear model
joblib.dump(linear_pipeline, os.path.join(OUT_DIR, 'linear_svm_pipeline.joblib'))
print("Saved linear_svm_pipeline.joblib")

# ---------------- SVM (RBF) with GridSearchCV ----------------
rbf_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE))
])

param_grid = {
    'clf__C': [0.1, 1, 10, 100],
    'clf__gamma': ['scale', 'auto', 0.01, 0.1, 1]
}

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(rbf_pipeline, param_grid, scoring='f1', n_jobs=-1, cv=cv, verbose=1)
print("\nRunning GridSearchCV for RBF SVM (this may take a while)...")
grid.fit(X_train, y_train)
print("GridSearch best params:", grid.best_params_)
best_rbf = grid.best_estimator_

# Evaluate best RBF
y_pred_rbf = best_rbf.predict(X_test)
y_proba_rbf = best_rbf.predict_proba(X_test)[:,1] if hasattr(best_rbf, "predict_proba") else None

cm_rbf = print_eval("RBF SVM (best)", y_test, y_pred_rbf)

plt.figure(figsize=(4,3))
sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='Greens')
plt.title('RBF SVM Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix_rbf.png'))
plt.close()
print("Saved confusion_matrix_rbf.png")

# Save RBF model
joblib.dump(best_rbf, os.path.join(OUT_DIR, 'rbf_svm_grid_pipeline.joblib'))
print("Saved rbf_svm_grid_pipeline.joblib")

# ---------------- ROC curve for RBF (if probabilities available) ----------------
if y_proba_rbf is not None:
    fpr, tpr, thresh = roc_curve(y_test, y_proba_rbf)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'RBF SVM (AUC = {roc_auc:.4f})')
    plt.plot([0,1],[0,1],'k--', linewidth=0.8)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - RBF SVM')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'roc_curve_rbf.png'))
    plt.close()
    print("Saved roc_curve_rbf.png (AUC = %.4f)" % roc_auc)
else:
    print("RBF pipeline has no predict_proba; ROC not available.")

# ---------------- Decision boundary plotting (2D) ----------------
# Use PCA to reduce features to 2D for visualization if necessary
n_features = len(numeric_cols) + sum([len(getattr(preprocessor, 'transformers_', []))])  # not used for logic, just info
# We'll create 2D representation using PCA if features > 2
from sklearn.pipeline import make_pipeline
# Fit preprocessor on full data so PCA sees same scaling/encoding
preprocessor_fit = preprocessor.fit(X_full)
X_processed = preprocessor_fit.transform(X_full)
# X_processed may be numpy array
if X_processed.shape[1] >= 2:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(X_processed)
    # Train a KNN-like plotting classifier using best_rbf's decision function via retrained RBF on 2D projection
    # Refit classifier on 2D for plotting regions
    clf_for_plot = SVC(kernel='rbf', C=grid.best_params_['clf__C'], gamma=grid.best_params_['clf__gamma'], probability=False, random_state=RANDOM_STATE)
    # split X2 correspondingly
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_full)
    clf_for_plot.fit(X2_train, y2_train)
    # create mesh
    x_min, x_max = X2[:,0].min()-1, X2[:,0].max()+1
    y_min, y_max = X2[:,1].min()-1, X2[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid_pts = np.c_[xx.ravel(), yy.ravel()]
    Z = clf_for_plot.predict(grid_pts)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
    scatter = plt.scatter(X2_test[:,0], X2_test[:,1], c=y2_test, cmap='coolwarm', edgecolor='k', s=40)
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.title('Decision regions (PCA 2D) - RBF SVM (plotted on 2D projection)')
    plt.legend(handles=scatter.legend_elements()[0], labels=list(np.unique(y_full).astype(str)), title="Classes")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'decision_regions_rbf_pca2d.png'), dpi=150)
    plt.close()
    print("Saved decision_regions_rbf_pca2d.png")
else:
    print("Not enough processed features to plot 2D decision boundary.")

# ---------------- Save summary CSV ----------------
summary = {
    'model': ['linear_svm', 'rbf_svm'],
    'accuracy': [accuracy_score(y_test, y_pred_linear), accuracy_score(y_test, y_pred_rbf)],
    'precision': [precision_score(y_test, y_pred_linear, zero_division=0), precision_score(y_test, y_pred_rbf, zero_division=0)],
    'recall': [recall_score(y_test, y_pred_linear, zero_division=0), recall_score(y_test, y_pred_rbf, zero_division=0)],
    'f1': [f1_score(y_test, y_pred_linear, zero_division=0), f1_score(y_test, y_pred_rbf, zero_division=0)]
}
pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, 'svm_model_comparison.csv'), index=False)
print("Saved svm_model_comparison.csv")

print("\nAll outputs saved to:", OUT_DIR)
print("Done.")
