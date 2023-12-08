import data_extraction
import embeddings_extraction
import results_extraction
import display_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import csv

class SVMClassifier:
    def __init__(self, input_dim=None):
        self.model = SVC()
        self.results = {'classification_report': None}

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        # Record overall metrics
        y_test_pred = self.model.predict(X_test)
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        self.results['classification_report'] = test_report

    def predict(self, input_embedding):
        # Make a prediction using the trained model
        return self.model.predict([input_embedding])

    def save_results(self, filename):
        # Save the overall metrics as a CSV file
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
            for class_label, metrics in self.results['classification_report'].items():
                if class_label.isnumeric():
                    writer.writerow([class_label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']])

        print(f"Results saved to {filename}")

def train_and_save_classifier(classifier, X, y, file_suffix):
    model = classifier()
    model.train(X, y)

    # Save the results
    filename = f'results/yelp_full_{file_suffix}.csv'
    model.save_results(filename)

    # Display the classification report
    print("Classification Report:")
    print(model.results['classification_report'])

    return model  # Return the trained model

def load_and_process_embeddings(file_path, column_name):
    df = pd.read_pickle(file_path)
    df[column_name] = df[column_name].apply(np.array)
    X = np.array(df[column_name].tolist())
    y = df['labels'].values

    return X, y

def main():
    # load the model and get the metrics

    embeddings_list = [('BART', 'bart_embeddings')]

    classifiers = [
        (SVMClassifier, 'svm'),
    ]

    trained_models = []

    for embedding_type, column_name in embeddings_list:
        file_path = f'embeddings/yelp_{embedding_type}_embeddings.pkl'
        X, y = load_and_process_embeddings(file_path, column_name)

        for classifier, suffix in classifiers:
            trained_model = train_and_save_classifier(classifier, X, y, f'{embedding_type}_{suffix}')
            trained_models.append((trained_model, embedding_type))
    
    print("How the model will be used to make Predictions")

    # review recommendation for restaurant with id 'M0c99tzIJPIbrY_RAO7KSQ'

    file_path = 'embeddings/yelp_BART_embeddings.pkl'

    # Read the pickle file into a DataFrame
    df = pd.read_pickle(file_path)

    # Specify the business ID you want to filter
    target_business_id = 'M0c99tzIJPIbrY_RAO7KSQ'

    # Create a new DataFrame with only rows that match the specified business ID
    filtered_df = df[df['business_id'] == target_business_id]

    # Display the new DataFrame
    print(filtered_df.head())

    # Calculate the average of the 'bart_embeddings' column
    average_embedding = np.mean(np.stack(filtered_df['bart_embeddings']), axis=0)

    print("Average Embedding:", average_embedding)

    # Make predictions using the trained models
    for trained_model, embedding_type in trained_models:
        prediction = trained_model.predict(average_embedding)
        print(f"Prediction using {embedding_type} model:", prediction)
        if prediction[0] == 1:
            print("Really don't go to this restaurant!")
        elif prediction[0] == 2:
            print("I would advise against it.")
        elif prediction[0] == 3:
            print("It's average.")
        elif prediction[0] == 4:
            print("I would advise for it.")
        elif prediction[0] == 5:
            print("Go to this restaurant!")

if __name__ == "__main__":
    main()
