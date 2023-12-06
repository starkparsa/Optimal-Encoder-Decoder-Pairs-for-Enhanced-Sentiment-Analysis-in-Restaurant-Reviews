import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

class GradientBoostingClassifierWrapper:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
        self.results = {'classification_report': []}

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training the Gradient Boosting model
        self.model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = self.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Save results
        self.results['classification_report'].append(report)

    def predict(self, X):
        return self.model.predict(X)

    def save_results(self, filename):
        # Convert the results to a DataFrame
        results_df = pd.DataFrame(self.results)

        # Save the DataFrame to a CSV file
        results_df.to_csv(filename, index=False)

        print(f"Results saved to {filename}")
