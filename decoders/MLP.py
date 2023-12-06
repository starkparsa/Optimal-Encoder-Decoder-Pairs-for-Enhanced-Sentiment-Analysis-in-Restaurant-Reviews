import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

class MLPClassifier:
    def __init__(self, input_dim, output_dim=1):
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_dim=input_dim))  # You can adjust the number of neurons in the hidden layer
        self.model.add(Dense(32, activation='relu'))  # You can add more hidden layers if needed
        self.model.add(Dense(output_dim, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.results = {'classification_report': []}

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training the model
        self.model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

        # Make predictions on the test set
        y_pred_prob = self.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)

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
