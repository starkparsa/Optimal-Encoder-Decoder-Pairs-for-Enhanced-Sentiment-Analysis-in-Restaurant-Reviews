# import data_extraction
# import embeddings_extraction
# import results_extraction
# import display_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from decoders.svm import SVMClassifier
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import inspect
import matplotlib.pyplot as plt

def train_and_save_classifier(classifier, X, y, file_suffix):
    if 'input_dim' in inspect.signature(classifier.__init__).parameters:
        input_dim = X.shape[1]
        model = classifier(input_dim=input_dim)
    else:
        model = classifier()

    model.train(X, y)
    model.save_results(f'results/yelp_{file_suffix}.pkl')

    # Generate and save plots
    accuracy_plot_path = f'results/yelp_{file_suffix}_accuracy.png'

    plt.figure(figsize=(12, 6))

    # Calculate accuracy directly using scikit-learn
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot([accuracy], label='Accuracy')
    plt.title('Accuracy Over Training')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot loss (commented out for SVM)
    # plt.subplot(1, 2, 2)
    # plt.plot(loss_values, label='Loss')
    # plt.title('Loss Over Training')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)

    # Save plots
    plt.tight_layout()
    plt.savefig(accuracy_plot_path)
    # plt.savefig(loss_plot_path)  # Commented out for SVM
    plt.close()

def load_and_process_embeddings(file_path, column_name):
    df = pd.read_pickle(file_path)
    df[column_name] = df[column_name].apply(lambda x: np.array(x))
    X = np.array(df[column_name].tolist())
    y = df['labels'].values

    return X, y

def main():
    embeddings_list = [('BART', 'bart_embeddings')]

    classifiers = [
        (SVMClassifier, 'svm'),
    ]

    for embedding_type, column_name in embeddings_list:
        file_path = f'embeddings/yelp_{embedding_type}_embeddings.pkl'
        X, y = load_and_process_embeddings(file_path, column_name)

        for classifier, suffix in classifiers:
            train_and_save_classifier(classifier, X, y, f'{embedding_type}_{suffix}')

if __name__ == "__main__":
    main()
