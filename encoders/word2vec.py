import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load the processed dataset from pickle
with open('processed_IMDB_Dataset.pkl', 'rb') as pickle_file:
    df = pickle.load(pickle_file)

from gensim.models import Word2Vec

# Tokenize and remove stopwords
stop_words = set(stopwords.words('english'))
df['tokenized_review'] = df['review'].apply(lambda x: [word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words])

# Train Word2Vec model
model = Word2Vec(sentences=df['tokenized_review'], vector_size=100, window=5, min_count=1, workers=4)

# Save the Word2Vec model
model.save('word2vec_model.bin')

# Load the Word2Vec model
model = Word2Vec.load('word2vec_model.bin')

# Example: Get the vector representation for the word 'good'
word_embedding = model.wv['good']
print(word_embedding)



