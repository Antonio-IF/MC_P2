# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Credit Score Prediction using Neural Network with GPU and AUC-ROC Plotting                  -- #
# -- script: model.py : python script with the model functionality using a powerful MLP and GPU support   -- #
# -- author: YOUR GITHUB USER NAME                                                                         -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                                 -- #
# -- repository: YOUR REPOSITORY URL                                                                       -- #
# -- --------------------------------------------------------------------------------------------------- -- #

# Required libraries
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

class CreditScoreModel:
    def __init__(self, data_path, batch_size=64, num_epochs=50, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.scaler = StandardScaler()
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.losses = []

    def load_data(self):
        """Load and split data into training and testing sets."""
        data_full = pd.read_excel(self.data_path)
        train_data, test_data = train_test_split(data_full, test_size=0.25, random_state=42)

        X_train = train_data.drop(columns=['Credit_Score'])
        y_train = train_data['Credit_Score']
        X_test = test_data.drop(columns=['Credit_Score'])
        y_test = test_data['Credit_Score']

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(self.device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.X_test_tensor = X_test_tensor
        self.y_test_tensor = y_test_tensor
        self.y_test = y_test

    def define_model(self, input_dim, output_dim):
        """Define the model architecture."""
        class CreditScoreNN(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(CreditScoreNN, self).__init__()
                self.fc1 = nn.Linear(input_dim, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, output_dim)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        self.model = CreditScoreNN(input_dim, output_dim).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train_model(self):
        """Train the neural network."""
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(self.train_loader)
            self.losses.append(epoch_loss)
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}')

    def evaluate_model(self):
        """Evaluate the model and calculate AUC-ROC."""
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(self.X_test_tensor)
            y_pred_proba = nn.functional.softmax(test_outputs, dim=1).cpu().numpy()
        
        y_test_bin = label_binarize(self.y_test, classes=[0, 1, 2])
        fpr, tpr, roc_auc = {}, {}, {}

        for i in range(len(np.unique(self.y_test))):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC Curve
        plt.figure(figsize=(8, 6))
        colors = ['blue', 'green', 'red']
        for i, color in zip(range(len(roc_auc)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Multi-class Classification')
        plt.legend(loc='lower right')
        plt.show()

    def plot_loss(self):
        """Plot the training loss over epochs."""
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, self.num_epochs+1), self.losses, label='Training Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.show()

    def save_model(self, filepath):
        """Save the trained model to a file."""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath, input_dim, output_dim):
        """Load a trained model from a file."""
        self.define_model(input_dim, output_dim)
        self.model.load_state_dict(torch.load(filepath))
        print(f"Model loaded from {filepath}")

    def final_evaluation(self):
        """Final evaluation on the test set."""
        _, y_pred_tensor = torch.max(self.model(self.X_test_tensor), 1)
        y_pred = y_pred_tensor.cpu().numpy()

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

# Call Class 
model = CreditScoreModel(data_path='Data/clean_data.xlsx')
model.load_data()  # Load the data
input_dim = model.X_test_tensor.shape[1]  # Access input dimension from loaded data

# Define the model using input_dim
model.define_model(input_dim=input_dim, output_dim=3)

# Train, evaluate, plot, save and load the model
model.train_model()
model.evaluate_model()
model.plot_loss()
model.save_model('credit_score_model.pth')
model.load_model('credit_score_model.pth', input_dim=input_dim, output_dim=3)
model.final_evaluation()
