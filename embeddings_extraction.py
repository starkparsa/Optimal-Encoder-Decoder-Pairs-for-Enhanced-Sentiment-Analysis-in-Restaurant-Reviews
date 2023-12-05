import pandas as pd
from encoders.BART import BartEmbedder
from encoders.BERT import BertEmbedder
from encoders.T5 import T5Embedder
from encoders.word2vec import Word2VecEmbedder

file_path = 'data/yelp_processed.pkl'

# List of embedders
embedders = [
    ('BART', BartEmbedder),
    ('BERT', BertEmbedder),
    ('T5', T5Embedder),
    ('Word2Vec', Word2VecEmbedder),
]

for model_name, embedder_class in embedders:
    # Read the pickle file into a DataFrame
    df = pd.read_pickle(file_path)

    # Initialize the embedder
    embedder = embedder_class()

    # Apply the model to each review and create a new column for embeddings
    df[f'{model_name.lower()}_embeddings'] = df['review'].apply(embedder.get_embeddings)

    # Save the processed data to a pickle file
    df.to_pickle(f'embeddings/yelp_{model_name}_embeddings.pkl')

    # Display the first few rows of the processed dataset
    print(df.head())
    print(f"Generated {model_name} embeddings\n{'='*40}\n")
