import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

class RandomForestClassifierWrapper:
    def __init__(self, n_estimators=30, max_depth=3, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.results = {'overall_metrics': None}

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training the Random Forest model
        self.model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = self.predict(X_test)

        # Evaluate the model
        report = classification_report(y_test, y_pred, output_dict=True)

        # Save results
        self.results['overall_metrics'] = report

    def predict(self, X):
        return self.model.predict(X)

    def save_results(self, filename):
        # Save only the overall metrics as a pickle file
        with open(filename, 'wb') as file:
            pickle.dump(self.results['overall_metrics'], file)

        print(f"Results saved to {filename}")
