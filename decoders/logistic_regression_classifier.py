import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class LogisticRegressionClassifier:
    def __init__(self, max_iter=10000):
        self.model = LogisticRegression(max_iter=max_iter)
        self.scaler = StandardScaler()  
        self.results = {'classification_report': []}

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)


       # Train the model on scaled data
        self.model.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = self.predict(X_test_scaled)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Save results
        self.results['classification_report'].append(report)

    def predict(self, X):
        return self.model.predict(X)

    def save_results(self, filename='logistic_regression_results.joblib'):
        
        # Convert the results to a DataFrame
        results_df = pd.DataFrame(self.results)

        # Save the DataFrame to a CSV file
        results_df.to_csv(filename, index=False)

        print(f"Results saved to {filename}")

