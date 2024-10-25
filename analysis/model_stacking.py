# --------------------------------------------------------------------------------------------------- #
# -- project: Credit Score Prediction using Stacking Model with Classic Models and GPU               -- #
# -- script: model.py : python script with the model functionality                                   -- #
# -- author: YOUR GITHUB USER NAME                                                                   -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                           -- #
# -- repository: YOUR REPOSITORY URL                                                                 -- #
# --------------------------------------------------------------------------------------------------- #

# Importing Libraries
import os
import pickle
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

# Ensure Models folder exists
os.makedirs('Models', exist_ok=True)

# Load and Split Data
def load_data(file_path):
    print("Loading data...")
    try:
        data = pd.read_excel(file_path)
        print(f"Data loaded successfully from {file_path}.")
        logging.info(f"Data loaded successfully from {file_path}")
        
        # Limit the data to 90%
        limited_data = data.sample(frac=0.90, random_state=42)
        print(f"Using 90% of the dataset, {len(limited_data)} rows.")
        
        print("Splitting data into training and testing sets...")
        train_data, test_data = train_test_split(limited_data, test_size=0.30, random_state=42)
        print("Data split completed.")
        logging.info("Data split into train and test sets")
        return train_data, test_data
    except Exception as e:
        print(f"Error loading data: {e}")
        logging.error(f"Error loading data: {e}")
        raise

# Preprocessing
def preprocess_data(train_data, test_data):
    print("Preprocessing data...")
    
    X_train = train_data.drop(columns=["Credit_Score"])
    y_train = train_data["Credit_Score"]
    X_test = test_data.drop(columns=["Credit_Score"])
    y_test = test_data["Credit_Score"]

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Data preprocessing completed.")
    logging.info("Data preprocessing completed")
    return X_train, X_test, y_train, y_test

# Train Models with Hyperparameter Optimization
def train_models_with_optimization(X_train, y_train):
    print("Training models with optimized hyperparameters...")

    # RandomForest with fewer trees
    rf_clf = RandomForestClassifier(n_estimators=50, 
                                    criterion="gini"
                                    max_depth=5, 
                                    random_state=42)

    # LogisticRegression with fewer iterations
    log_clf = LogisticRegression(max_iter=100, random_state=42)

    # Replace LinearSVC with SVC to support predict_proba
    svc_clf = SVC(max_iter=1000, random_state=42, probability=True)

    # Combining all models in a VotingClassifier with soft voting
    voting_clf = VotingClassifier(
        estimators=[('rf', rf_clf), ('log_reg', log_clf), ('svc', svc_clf)],
        voting='soft'  # Switch to soft voting
    )

    # Cross-validation to evaluate ensemble model
    scores = cross_val_score(voting_clf, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
    print(f"Cross-validated accuracy scores: {scores}")
    logging.info(f"Cross-validated accuracy scores: {scores}")

    # Fit the model on the entire training set
    print("Fitting the model on the training data...")
    voting_clf.fit(X_train, y_train)
    print("Model training completed.")
    logging.info("Voting Classifier trained successfully")

    return voting_clf

# Plot AUC-ROC Curve for Multi-class Classification
def plot_roc_auc(model, X_test, y_test):
    print("Plotting AUC-ROC...")
    # Binarize the output (required for multi-class ROC AUC)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # Assuming 3 classes, adjust accordingly
    y_score = model.predict_proba(X_test)

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(y_test_bin.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_score[:, i])

    plt.figure()
    for i in range(y_test_bin.shape[1]):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

# Plot Confusion Matrix
def plot_confusion_matrix(model, X_test, y_test):
    print("Plotting Confusion Matrix...")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

# Model Evaluation and Saving (with plotting)
def evaluate_and_save_model(model, X_test, y_test):
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Model Accuracy: {accuracy}")
    print(f"Model F1 Score: {f1}")
    
    logging.info(f"Model Accuracy: {accuracy}")
    logging.info(f"Model F1 Score: {f1}")
    
    # Save the model using pickle
    model_path = 'Models/Best_stackingmodel.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved successfully to {model_path}.")
    logging.info(f"Model saved to {model_path}")
    
    # Plot Confusion Matrix
    plot_confusion_matrix(model, X_test, y_test)
    
    # Plot AUC-ROC
    plot_roc_auc(model, X_test, y_test)
    
    return accuracy, f1

# Main pipeline function
def run_pipeline(file_path):
    print("Starting the pipeline...")
    
    # Load and preprocess data
    train_data, test_data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(train_data, test_data)
    
    # Train models
    model = train_models_with_optimization(X_train, y_train)
    
    # Evaluate and save the model
    accuracy, f1 = evaluate_and_save_model(model, X_test, y_test)
    
    print("Pipeline completed.")
    return accuracy, f1

# Path to the dataset (to be passed as argument)
data_path = "Data/clean_data.xlsx"

# Example run (this is where the model is trained and evaluated)
accuracy, f1 = run_pipeline(data_path)
print(f"Accuracy: {accuracy}, F1 Score: {f1}")
