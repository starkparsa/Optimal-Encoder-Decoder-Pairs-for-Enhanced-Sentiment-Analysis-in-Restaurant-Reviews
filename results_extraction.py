import pandas as pd
from decoders.svm import SVMClassifier
from decoders.logistic_regression_classifier import LogisticRegressionClassifier
import numpy as np
import pickle

# Load data from a pickle file
file_path = 'embddings/Word2Ve_base_IMDB_Embeddings.pkl'
df = pd.read_pickle(file_path)



# Ensure 'Word2Vec_embeddings' contains NumPy arrays directly
df['Word2Vec_embeddings'] = df['Word2Vec_embeddings'].apply(lambda x: np.array(x))

# Create a list to store the NumPy arrays
x = []

# Extract NumPy arrays from the DataFrame and append to the list
for v in df['Word2Vec_embeddings']:
    x.append(v)

# Assuming your DataFrame has columns 'Word2Vec_embeddings' and 'sentiment'
X = x  # Use the list 'x' instead of the DataFrame column directly
y = df['sentiment']








# Create and train the SVM classifier
svm_classifier = SVMClassifier()
svm_classifier.train(X, y)

# Save the results for future analysis
svm_classifier.save_results('results/IMDB_Word2Ve_base_svm.csv')





logistic_classifier = LogisticRegressionClassifier()
logistic_classifier.train(X, y)

# Save the results for future analysis
logistic_classifier.save_results('results/IMDB_Word2Ve_base_logistic_regression_classifier.csv')




