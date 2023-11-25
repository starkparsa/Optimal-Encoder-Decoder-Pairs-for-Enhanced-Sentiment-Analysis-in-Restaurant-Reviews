# encoders/Word2Vec.py

import spacy
import numpy as np

class Word2VecEmbedder:
    def __init__(self, model_name='en_core_web_md'):
        self.nlp = spacy.load(model_name)

    def get_embeddings(self, text):
        doc = self.nlp(text)
        review_embedding = np.zeros(self.nlp.vocab.vectors.shape[1])  # Initialize with zeros
        count = 0  # Number of words with vectors in the review

        for word in doc:
            if word.has_vector:
                review_embedding += word.vector
                count += 1

        if count > 0:
            return review_embedding / count
        else:
            return review_embedding


