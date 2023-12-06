import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

class NaiveBayesClassifier:
    def __init__(self):
        self.model = MultinomialNB()
        self.results = {'classification_report': []}

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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

    def save_results(self, filename='naive_bayes_results.joblib'):
        joblib.dump(self.results, filename)

    
    