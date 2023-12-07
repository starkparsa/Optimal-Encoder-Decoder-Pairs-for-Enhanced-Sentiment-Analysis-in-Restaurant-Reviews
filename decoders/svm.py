from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

class SVMClassifier:
    def __init__(self, input_dim=None):
        self.model = SVC()
        self.results = {'overall_metrics': None}

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        self.results['overall_metrics'] = report

    def predict(self, X):
        return self.model.predict(X)
    
    def save_results(self, filename):
        # Save only the overall metrics as a pickle file
        with open(filename, 'wb') as file:
            pickle.dump(self.results['overall_metrics'], file)

        print(f"Results saved to {filename}")