
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: model.py : python script with the model functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import torch # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 to activate paralellize
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0)) # NVIDIA GeForce RTX 3060 Ti

# Load dataset:
data_full = pd.read_excel('Data/clean_data.xlsx')


# Dividir el dataset en entrenamiento (75%) y prueba (25%)
train_data, test_data = train_test_split(data_full, test_size=0.25, random_state=42)

# Prepare the dataset: Features and target
X_train = train_data.drop(columns=['Credit_Score'])
y_train = train_data['Credit_Score']

X_test = test_data.drop(columns=['Credit_Score'])
y_test = test_data['Credit_Score']

# Step 1: Feature Selection using RandomForest for importance-based filtering
# Train a simple RandomForest to identify important features
rf_feature_selector = RandomForestClassifier(n_estimators=100, random_state=0)
rf_feature_selector.fit(X_train, y_train)

# Get feature importances and sort them in descending order
importances = rf_feature_selector.feature_importances_
indices = importances.argsort()[::-1]

# Print the most important features
print("Feature ranking based on RandomForest importance:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. Feature {X_train.columns[indices[f]]} ({importances[indices[f]]:.4f})")

# Step 2: Recursive Feature Elimination (RFE) for optimal feature set
rfe_selector = RFE(estimator=rf_feature_selector, n_features_to_select=10, step=1)
rfe_selector = rfe_selector.fit(X_train, y_train)

# Print selected features
print("Selected features by RFE:")
selected_features = X_train.columns[rfe_selector.support_]
print(selected_features)

# Filter the dataset to include only the selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Step 3: Apply standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Step 4: Define base models for stacking
estimators = [
    ('rf', RandomForestClassifier(n_estimators=120, max_depth=10, random_state=0)),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=0)),
    ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, eval_metric='logloss', random_state=0, use_label_encoder=False, tree_method='gpu_hist')),  # Use GPU
    ('ridge', RidgeClassifier(alpha=1.0))
]

# Meta-model: Logistic Regression
stacking_model = StackingClassifier(
    estimators=estimators, 
    final_estimator=LogisticRegression(max_iter=5000, solver='lbfgs')
)

# Step 5: Hyperparameter tuning with GridSearchCV
param_grid_stacking = {
    'final_estimator__C': [0.1, 1],
    'rf__n_estimators': [80, 120],
    'rf__max_depth': [5, 10],
    'gb__learning_rate': [0.05],
    'xgb__learning_rate': [0.05],
    'xgb__max_depth': [3]
}

# Optimize for F1 score using 3-fold cross-validation
stacking_grid = GridSearchCV(stacking_model, param_grid_stacking, scoring='f1', refit='f1', cv=3, n_jobs=-1)
stacking_grid.fit(X_train_scaled, y_train)

# Best model from GridSearch
best_stacking_model = stacking_grid.best_estimator_

# Step 6: Predictions and evaluation for the stacking model
y_pred_stacking = best_stacking_model.predict(X_test_scaled)
f1_stacking = f1_score(y_test, y_pred_stacking)
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
classification_report_stacking = classification_report(y_test, y_pred_stacking)

# AUC-ROC for the stacking model
y_probs_stacking = best_stacking_model.predict_proba(X_test_scaled)[:, 1]
fpr_stacking, tpr_stacking, _ = roc_curve(y_test, y_probs_stacking)
auc_stacking = roc_auc_score(y_test, y_probs_stacking)

# Confusion Matrix for the stacking model
conf_matrix_stacking = confusion_matrix(y_test, y_pred_stacking)

# Step 7: Train a simple Logistic Regression model for comparison
logistic_model_simple = LogisticRegression(max_iter=5000, solver='lbfgs')
logistic_model_simple.fit(X_train_scaled, y_train)

# Predictions and evaluation for Logistic Regression
y_pred_logistic = logistic_model_simple.predict(X_test_scaled)
f1_logistic = f1_score(y_test, y_pred_logistic)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
classification_report_logistic = classification_report(y_test, y_pred_logistic)

# AUC-ROC for Logistic Regression
y_probs_logistic = logistic_model_simple.predict_proba(X_test_scaled)[:, 1]
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_probs_logistic)
auc_logistic = roc_auc_score(y_test, y_probs_logistic)

# Confusion Matrix for Logistic Regression
conf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)

# Step 8: Plot ROC Curves for both models
plt.figure(figsize=(10, 6))
plt.plot(fpr_logistic, tpr_logistic, label=f'Logistic Regression (AUC = {auc_logistic:.2f})')
plt.plot(fpr_stacking, tpr_stacking, label=f'Stacking Model (AUC = {auc_stacking:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison: Logistic Regression vs Stacking Model')
plt.legend()
plt.grid(True)
plt.show()

# Step 9: Print classification reports
print("Logistic Regression Classification Report:")
print(classification_report_logistic)

print("Stacking Model Classification Report:")
print(classification_report_stacking)

# Step 10: Plot confusion matrices for both models
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_logistic, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_stacking, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Stacking Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.tight_layout()
plt.show()