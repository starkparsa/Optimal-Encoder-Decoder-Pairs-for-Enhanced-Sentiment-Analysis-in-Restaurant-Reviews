# import pandas as pd
# from encoders.BART import BartEmbedder
# from encoders.BERT import BertEmbedder
# from encoders.T5 import T5Embedder
# from encoders.word2vec import Word2VecEmbedder

# file_path = 'data/yelp_processed.pkl'

# # List of embedders
# embedders = [
#     ('BART', BartEmbedder),
#     ('BERT', BertEmbedder),
#     ('T5', T5Embedder),
#     ('Word2Vec', Word2VecEmbedder),
# ]

# for model_name, embedder_class in embedders:
#     # Read the pickle file into a DataFrame
#     df = pd.read_pickle(file_path)

#     # Initialize the embedder
#     embedder = embedder_class()

#     # Apply the model to each review and create a new column for embeddings
#     df[f'{model_name.lower()}_embeddings'] = df['review'].apply(embedder.get_embeddings)

#     # Save the processed data to a pickle file
#     df.to_pickle(f'embeddings/yelp_{model_name}_embeddings.pkl')

#     # Display the first few rows of the processed dataset
#     print(df.head())
#     print(f"Generated {model_name} embeddings\n{'='*40}\n")




import pandas as pd
from concurrent.futures import ThreadPoolExecutor  # Use ThreadPoolExecutor for IO-bound tasks

from encoders.BART import BartEmbedder
from encoders.BERT import BertEmbedder
from encoders.T5 import T5Embedder
from encoders.word2vec import Word2VecEmbedder

def process_model(model_name, embedder_class, df):
    embedder = embedder_class()
    df[f'{model_name.lower()}_embeddings'] = df['review'].apply(embedder.get_embeddings)
    df.to_pickle(f'embeddings/yelp_{model_name}_embeddings.pkl')
    print(f"Generated {model_name} embeddings\n{'='*40}\n")

def main():
    file_path = 'data/yelp_processed.pkl'

    # Read the pickle file into a DataFrame once
    df = pd.read_pickle(file_path)

    # List of embedders
    embedders = [
        ('BART', BartEmbedder),
        ('BERT', BertEmbedder),
        ('T5', T5Embedder),
        ('Word2Vec', Word2VecEmbedder),
    ]

    # Use ThreadPoolExecutor for IO-bound tasks
    with ThreadPoolExecutor() as executor:
        # Process each model in parallel
        futures = [executor.submit(process_model, name, embedder, df.copy()) for name, embedder in embedders]

        # Wait for all tasks to complete
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()
