# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Credit Score Prediction using Neural Network with GPU and AUC-ROC Plotting                  -- #
# -- script: model.py : python script with the model functionality using a powerful MLP and GPU support   -- #
# -- author: YOUR GITHUB USER NAME                                                                         -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                                 -- #
# -- repository: YOUR REPOSITORY URL                                                                       -- #
# -- --------------------------------------------------------------------------------------------------- -- #

# Required libraries
import os
import pickle
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
print(tf.__version__)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

class CreditScoreModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train_cat = None
        self.y_test_cat = None
        self.y_test = None

    def load_data(self):
        """Load data and split it into training and testing sets."""
        print("Loading data...")
        data = pd.read_excel(self.file_path)
        limited_data = data.sample(frac=0.90, random_state=42)
        train_data, test_data = train_test_split(limited_data, test_size=0.30, random_state=42)
        return train_data, test_data

    def preprocess_data(self, train_data, test_data):
        """Preprocess the data by standardizing features and binarizing labels."""
        X_train = train_data.drop(columns=["Credit_Score"])
        y_train = train_data["Credit_Score"]
        X_test = test_data.drop(columns=["Credit_Score"])
        y_test = test_data["Credit_Score"]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train_cat = to_categorical(y_train, num_classes=3)
        y_test_cat = to_categorical(y_test, num_classes=3)

        self.X_train, self.X_test = X_train, X_test
        self.y_train_cat, self.y_test_cat = y_train_cat, y_test_cat
        self.y_test = y_test

    def build_tuned_model(self, hp):
        """Build a model with tunable hyperparameters."""
        model = Sequential()
        
        # Input layer
        model.add(Dense(hp.Int('units_input', min_value=64, max_value=256, step=64), activation='relu', input_shape=(self.X_train.shape[1],)))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_input', min_value=0.2, max_value=0.5, step=0.1)))
        
        # Hidden layers
        for i in range(hp.Int('num_layers', 2, 4)):
            model.add(Dense(hp.Int(f'units_{i}', min_value=32, max_value=128, step=32), activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.5, step=0.1)))

        # Output layer
        model.add(Dense(3, activation='softmax'))
        
        model.compile(
            optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def tune_model(self):
        """Tune the model using Keras Tuner's Hyperband."""
        tuner = Hyperband(
            self.build_tuned_model,
            objective='val_accuracy',
            max_epochs=50,
            factor=3,
            directory='hyperband_tuning',
            project_name='credit_score_nn_tuning'
        )
        
        tuner.search(self.X_train, self.y_train_cat, epochs=50, validation_data=(self.X_test, self.y_test_cat), batch_size=32)
        
        self.model = tuner.get_best_models(num_models=1)[0]
        tuner.results_summary()

    def plot_loss(self, history):
        """Plot the loss curve."""
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

    def plot_roc_auc(self):
        """Plot the AUC-ROC curve for multi-class classification."""
        y_test_bin = label_binarize(self.y_test, classes=[0, 1, 2])
        y_score = self.model.predict(self.X_test)

        fpr = {}
        tpr = {}
        roc_auc = {}

        plt.figure()
        for i in range(y_test_bin.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_score[:, i])
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()

    def plot_confusion_matrix(self):
        """Plot the confusion matrix."""
        y_pred = np.argmax(self.model.predict(self.X_test), axis=1)
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

    def evaluate_model(self):
        """Evaluate the model and plot results."""
        y_pred = np.argmax(self.model.predict(self.X_test), axis=1)

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        print(f"Model Accuracy: {accuracy}")
        print(f"Model F1 Score: {f1}")

        # Save the model
        self.model.save('Models/NeuralNetwork_BestModel.h5')
        print("Model saved successfully.")
        
        self.plot_confusion_matrix()
        self.plot_roc_auc()

        return accuracy, f1

    def run(self):
        """Run the full pipeline of data loading, preprocessing, tuning, and evaluation."""
        train_data, test_data = self.load_data()
        self.preprocess_data(train_data, test_data)
        self.tune_model()
        return self.evaluate_model()

data_path = "Data/clean_data.xlsx"
model = CreditScoreModel(data_path)
accuracy, f1 = model.run()
print(f"Accuracy: {accuracy}, F1 Score: {f1}")
