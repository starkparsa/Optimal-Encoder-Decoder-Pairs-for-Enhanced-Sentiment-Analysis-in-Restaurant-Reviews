import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import pickle

class MLPClassifier:
    def __init__(self, input_dim, output_dim=5):
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_dim=input_dim))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(output_dim, activation='sigmoid'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.results = {'overall_metrics': None}

    def train(self, X, y):
        # Convert labels to one-hot encoding
        y_one_hot = to_categorical(y - 1, num_classes=5)

        X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

        # Training the model
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

        # Make predictions on the test set
        y_pred_prob = self.predict(X_test)
        y_pred_labels = np.argmax(y_pred_prob, axis=1) + 1  # Shift labels to the range 1-5

        # Evaluate the model
        report = classification_report(y_test.argmax(axis=1) + 1, y_pred_labels, output_dict=True)

        # Save results
        self.results['overall_metrics'] = report

    def predict(self, X):
        return self.model.predict(X)

    def save_results(self, filename):
        # Save only the overall metrics as a pickle file
        with open(filename, 'wb') as file:
            pickle.dump(self.results['overall_metrics'], file)

        print(f"Results saved to {filename}")

# Example usage:
# mlp_classifier = MLPClassifier(input_dim=your_input_dim)
# mlp_classifier.train(X_train, y_train)
# mlp_classifier.save_results('mlp_results.pkl')
