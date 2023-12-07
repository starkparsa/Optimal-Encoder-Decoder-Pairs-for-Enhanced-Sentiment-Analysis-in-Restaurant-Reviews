import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

class CNNClassifier:
    def __init__(self, input_dim, filters=128, kernel_size=5, hidden_units=128, output_units=5):
        self.model = Sequential([
            Conv1D(filters, kernel_size, activation='relu', input_shape=(input_dim, 1)),
            GlobalMaxPooling1D(),
            Dense(hidden_units, activation='relu'),
            Dense(output_units, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.results = {'overall_metrics': None}

    def train(self, X, y, epochs=50, test_size=0.2, random_state=42):
        # Assuming X is a 3D array (number of samples, sequence length, features)
        X = np.expand_dims(X, axis=-1)  # Add a channel dimension

        # Convert labels to start from 0
        y -= 1

        # Convert labels to one-hot encoding
        y_one_hot = to_categorical(y, num_classes=5)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=test_size, random_state=random_state)

        # Train the model
        self.model.fit(X_train, y_train, epochs=epochs)

        # Make predictions on the test set
        y_pred = self.predict(X_test)

        # Convert one-hot encoded predictions back to original labels
        y_pred_labels = np.argmax(y_pred, axis=1) + 1

        # Evaluate the model
        report = classification_report(np.argmax(y_test, axis=1) + 1, y_pred_labels, output_dict=True)

        # Save results
        self.results['overall_metrics'] = report

    def predict(self, X):
        X = np.expand_dims(X, axis=-1)  # Add a channel dimension
        return self.model.predict(X)

    def save_results(self, filename):
        # Save only the overall metrics as a pickle file
        with open(filename, 'wb') as file:
            pickle.dump(self.results['overall_metrics'], file)

        print(f"Results saved to {filename}")


