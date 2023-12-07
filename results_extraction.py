import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from decoders.svm import SVMClassifier
from decoders.logistic_regression import LogisticRegressionClassifier
from decoders.CNN import CNNClassifier
from decoders.Gradient_Boosting import GradientBoostingClassifierWrapper
from decoders.MLP import MLPClassifier
from decoders.Random_Forest import RandomForestClassifierWrapper
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import inspect

def train_and_save_classifier(classifier, X, y, file_suffix):
    if 'input_dim' in inspect.signature(classifier.__init__).parameters:
        input_dim = X.shape[1]  # Assuming X is a 2D array where each row corresponds to a sample
        model = classifier(input_dim=input_dim)  # Pass the input_dim argument
    else:
        model = classifier()

    model.train(X, y)
    model.save_results(f'results/yelp_{file_suffix}.pkl')


def load_and_process_embeddings(file_path, column_name):
    df = pd.read_pickle(file_path)
    df[column_name] = df[column_name].apply(lambda x: np.array(x))
    X = np.array(df[column_name].tolist())
    y = df['labels'].values

    return X, y

def main():
    embeddings_list = [
        ('Word2Vec', 'word2vec_embeddings'),
        ('BERT', 'bert_embeddings'),
        ('BART', 'bart_embeddings'),
        ('T5', 't5_embeddings')
    ]

    classifiers = [
        # (LogisticRegressionClassifier, 'logistic_regression'),
        # (SVMClassifier, 'svm'),
        (MLPClassifier, 'mlp'),
        (CNNClassifier, 'cnn'),
        # (GradientBoostingClassifierWrapper, 'gradient_boosting'),
        # (RandomForestClassifierWrapper, 'random_forest')
    ]

    for embedding_type, column_name in embeddings_list:
        file_path = f'embeddings/yelp_{embedding_type}_embeddings.pkl'
        X, y = load_and_process_embeddings(file_path, column_name)

        for classifier, suffix in classifiers:
            train_and_save_classifier(classifier, X, y, f'{embedding_type}_{suffix}')

if __name__ == "__main__":
    main()
