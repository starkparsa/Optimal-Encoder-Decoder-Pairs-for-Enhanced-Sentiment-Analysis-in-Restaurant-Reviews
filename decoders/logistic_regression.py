import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import pickle

class LogisticRegressionClassifier:
    def __init__(self, input_dim=None):
        self.model = LogisticRegression()
        self.results = {'overall_metrics': None, 'accuracy': None, 'f1-score': None, 'precision': None, 'recall': None, 'weighted avg': None}

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save overall metrics
        self.results['overall_metrics'] = report

        # Calculate and save accuracy
        accuracy = accuracy_score(y_test, y_pred)
        self.results['accuracy'] = accuracy
        
        # Save f1-score, precision, recall, and weighted avg
        self.results['f1-score'] = report['weighted avg']['f1-score']
        self.results['precision'] = report['weighted avg']['precision']
        self.results['recall'] = report['weighted avg']['recall']
        self.results['weighted avg'] = report['weighted avg']

    def predict(self, X):
        return self.model.predict(X)
    
    def save_results(self, filename):
        # Save both overall metrics and accuracy as a pickle file
        with open(filename, 'wb') as file:
            pickle.dump(self.results, file)

        print(f"Results saved to {filename}")
