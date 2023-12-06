from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

class SVMClassifier:
    def __init__(self, input_dim=None):  # Update the constructor
        self.model = SVC()
        self.results = {'classification_report': []}

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred)
        self.results['classification_report'].append(report)

    def predict(self, X):
        return self.model.predict(X)
    
    def save_results(self, filename):
        # Convert the results to a DataFrame
        results_df = pd.DataFrame(self.results)

        # Save the DataFrame to a pickle file
        results_df.to_pickle(filename)

        print(f"Results saved to {filename}")