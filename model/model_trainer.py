import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from joblib import Parallel, delayed


class ModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {
            'Logistic Regression': LogisticRegression(),
            'SVM': SVC(kernel='linear'),
            'LDA': LinearDiscriminantAnalysis()
        }

    def train_model(self, model_name, model):
        print(f"Training {model_name}...")
        model.fit(self.X_train, self.y_train)

    def evaluate_model(self, model_name, model):
        y_pred = model.predict(self.X_test)
        
        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy of {model_name}: {accuracy}")
        
        # Classification Report
        report = classification_report(self.y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica'])
        print(f"Classification Report for {model_name}:\n{report}\n")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"Confusion Matrix for {model_name}:\n{cm}\n")

    def save_model(self, model_name, model):
        joblib.dump(model, f'model/saved_models/{model_name.lower().replace(" ", "_")}.pkl')
    
    def train_and_evaluate_all(self):
        Parallel(n_jobs=len(self.models))(
            delayed(self.train_and_evaluate)(name, model) for name, model in self.models.items()
        )

    def train_and_evaluate(self, model_name, model):
        self.train_model(model_name, model)
        self.evaluate_model(model_name, model)
        self.save_model(model_name, model)