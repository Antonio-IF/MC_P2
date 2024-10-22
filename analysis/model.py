"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Credit Score Prediction using Stacking Model with Early Stopping and GPU                     -- #
# -- script: model.py : python script with the model functionality including early stopping and GPU        -- #
# -- author: YOUR GITHUB USER NAME                                                                         -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                                 -- #
# -- repository: YOUR REPOSITORY URL                                                                       -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import torch
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier  # Use LightGBM for faster training
import seaborn as sns


# Step 1: Check GPU availability and print GPU details
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Running on CPU.")

# Load dataset
data_full = pd.read_excel('Data/clean_data.xlsx')

# Split the dataset into training (75%) and testing (25%)
train_data, test_data = train_test_split(data_full, test_size=0.25, random_state=42)

# Prepare the dataset: Features and target
X_train = train_data.drop(columns=['Credit_Score'])
y_train = train_data['Credit_Score']

X_test = test_data.drop(columns=['Credit_Score'])
y_test = test_data['Credit_Score']

# Step 2: Feature Selection using RandomForest for importance-based filtering
rf_feature_selector = RandomForestClassifier(n_estimators=100, random_state=0)
rf_feature_selector.fit(X_train, y_train)

# Get feature importances and sort them in descending order
importances = rf_feature_selector.feature_importances_
indices = importances.argsort()[::-1]

# Print the most important features
print("Feature ranking based on RandomForest importance:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. Feature {X_train.columns[indices[f]]} ({importances[indices[f]]:.4f})")

# Step 3: Recursive Feature Elimination (RFE) for optimal feature set
rfe_selector = RFE(estimator=rf_feature_selector, n_features_to_select=10, step=1)
rfe_selector = rfe_selector.fit(X_train, y_train)

# Print selected features
selected_features = X_train.columns[rfe_selector.support_]
print("Selected features by RFE:")
print(selected_features)

# Filter the dataset to include only the selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Step 4: Apply standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Step 5: Define base models for stacking with LightGBM
estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=8, random_state=0, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=5, random_state=0)),
    ('lgb', LGBMClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=0, device='gpu')),  # Use LightGBM with GPU
    ('ridge', RidgeClassifier(alpha=1.0))
]

# Meta-model: Logistic Regression or LightGBM
stacking_model = StackingClassifier(
    estimators=estimators, 
    final_estimator=LGBMClassifier(n_estimators=100, random_state=0, device='gpu')  # Use LightGBM as meta-model for better performance
)

# Step 6: Hyperparameter tuning with GridSearchCV
param_grid_stacking = {
    'rf__n_estimators': [25, 50],
    'rf__max_depth': [3, 5],
    'gb__learning_rate': [0.05],
    'lgb__learning_rate': [0.05],
    'lgb__max_depth': [3]
}

# Optimize for F1 score using 3-fold cross-validation for multiclass
stacking_grid = GridSearchCV(stacking_model, param_grid_stacking, scoring='f1_macro', refit='f1_macro', cv=3, n_jobs=-1)
stacking_grid.fit(X_train_scaled, y_train)

# Best model from GridSearch
best_stacking_model = stacking_grid.best_estimator_

# Step 7: Predictions and evaluation for the stacking model
y_pred_stacking = best_stacking_model.predict(X_test_scaled)
f1_stacking = f1_score(y_test, y_pred_stacking, average='macro')  # Use macro for multiclass
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
classification_report_stacking = classification_report(y_test, y_pred_stacking)

# AUC-ROC for the stacking model
y_probs_stacking = best_stacking_model.predict_proba(X_test_scaled)[:, 1]
fpr_stacking, tpr_stacking, _ = roc_curve(y_test, y_probs_stacking)
auc_stacking = roc_auc_score(y_test, y_probs_stacking)

# Confusion Matrix for the stacking model
conf_matrix_stacking = confusion_matrix(y_test, y_pred_stacking)

# Step 8: Train a simple LightGBM model for comparison
lgb_simple = LGBMClassifier(n_estimators=100, random_state=0, device='gpu')
lgb_simple.fit(X_train_scaled, y_train)

# Predictions and evaluation for LightGBM
y_pred_lgb = lgb_simple.predict(X_test_scaled)
f1_lgb = f1_score(y_test, y_pred_lgb, average='macro')
accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
classification_report_lgb = classification_report(y_test, y_pred_lgb)

# AUC-ROC for LightGBM
y_probs_lgb = lgb_simple.predict_proba(X_test_scaled)[:, 1]
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_probs_lgb)
auc_lgb = roc_auc_score(y_test, y_probs_lgb)

# Confusion Matrix for LightGBM
conf_matrix_lgb = confusion_matrix(y_test, y_pred_lgb)

# Step 9: Plot ROC Curves for both models
plt.figure(figsize=(10, 6))
plt.plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC = {auc_lgb:.2f})')
plt.plot(fpr_stacking, tpr_stacking, label=f'Stacking Model (AUC = {auc_stacking:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison: LightGBM vs Stacking Model')
plt.legend()
plt.grid(True)
plt.show()

# Step 10: Print classification reports
print("LightGBM Classification Report:")
print(classification_report_lgb)

print("Stacking Model Classification Report:")
print(classification_report_stacking)

# Step 11: Plot confusion matrices for both models
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_lgb, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: LightGBM')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_stacking, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Stacking Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.tight_layout()
plt.show()
