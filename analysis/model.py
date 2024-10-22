# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Credit Score Prediction using Stacking Model with Classic Models and GPU                     -- #
# -- script: model.py : python script with the model functionality including additional models, early stopping, and GPU -- #
# -- author: YOUR GITHUB USER NAME                                                                         -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                                 -- #
# -- repository: YOUR REPOSITORY URL                                                                       -- #
# -- --------------------------------------------------------------------------------------------------- -- #

# Required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Step 1: Check GPU availability
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Running on CPU.")

# Step 2: Load the dataset and split into train and test
data_full = pd.read_excel('Data/clean_data.xlsx')
train_data, test_data = train_test_split(data_full, test_size=0.25, random_state=42)

# Step 3: Prepare features and target
X_train = train_data.drop(columns=['Credit_Score'])
y_train = train_data['Credit_Score']

X_test = test_data.drop(columns=['Credit_Score'])
y_test = test_data['Credit_Score']

# Step 4: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Enhance Stacking model with more classic models
# Base models
estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=10, random_state=42)),
    ('svc', SVC(probability=True, random_state=42)),
    ('nb', GaussianNB()),
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('knn', KNeighborsClassifier())
]

# Meta-classifier
stacking_model = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression(), cv=5
)

# Step 6: Train the stacking model
stacking_model.fit(X_train_scaled, y_train)

# Step 7: Make predictions
y_pred = stacking_model.predict(X_test_scaled)
y_proba = stacking_model.predict_proba(X_test_scaled)

# Check if the target variable has multiple classes
num_classes = len(np.unique(y_test))

# Step 8: Drop AUC-ROC of different models
roc_auc_scores = {}
for name, model in estimators:
    model.fit(X_train_scaled, y_train)
    y_proba_model = model.predict_proba(X_test_scaled)
    
    if num_classes > 2:
        roc_auc_scores[name] = roc_auc_score(y_test, y_proba_model, multi_class='ovr')
    else:
        roc_auc_scores[name] = roc_auc_score(y_test, y_proba_model[:, 1])

# Print AUC-ROC scores of different models
print("AUC-ROC Scores for Base Models:")
for name, score in roc_auc_scores.items():
    print(f"{name}: {score:.4f}")

# AUC-ROC for the final stacking model
if num_classes > 2:
    roc_auc_stacking = roc_auc_score(y_test, y_proba, multi_class='ovr')
else:
    roc_auc_stacking = roc_auc_score(y_test, y_proba[:, 1])

print(f"AUC-ROC for Stacking Model: {roc_auc_stacking:.4f}")

# Step 9: Evaluate the stacking model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Step 10: Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step 11: Real vs Predicted Credit_Score comparison
comparison_df = pd.DataFrame({'Real Credit_Score': y_test, 'Predicted Credit_Score': y_pred})
print(comparison_df.head())

# Step 12: Learning curve (loss curve and accuracy over epochs)
train_sizes = np.linspace(0.1, 1.0, 10)
train_scores = []
test_scores = []

for frac in train_sizes:
    frac = float(frac)  # Convert np.float64 to standard Python float
    X_train_frac, _, y_train_frac, _ = train_test_split(X_train_scaled, y_train, train_size=frac, random_state=42)
    stacking_model.fit(X_train_frac, y_train_frac)
    train_pred = stacking_model.predict(X_train_frac)
    test_pred = stacking_model.predict(X_test_scaled)
    train_scores.append(accuracy_score(y_train_frac, train_pred))
    test_scores.append(accuracy_score(y_test, test_pred))

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores, label='Train Accuracy', marker='o')
plt.plot(train_sizes, test_scores, label='Test Accuracy', marker='o')
plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.show()

# Step 13: ROC curve for Stacking Model (for multi-class classification)
if num_classes > 2:
    fpr = {}
    tpr = {}
    roc_auc_dict = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_proba[:, i], pos_label=i)
        roc_auc_dict[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc_dict[i]:.4f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC curve for Stacking Model')
    plt.legend(loc='lower right')
    plt.show()

else:
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'AUC-ROC = {roc_auc_stacking:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Stacking Model')
    plt.legend()
    plt.show()
