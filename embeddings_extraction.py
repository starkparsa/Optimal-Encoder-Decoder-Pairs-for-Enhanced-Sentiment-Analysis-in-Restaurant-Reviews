import pandas as pd
import spacy
from concurrent.futures import ThreadPoolExecutor
from encoders.BART import BartEmbedder
from encoders.BERT import BertEmbedder
from encoders.T5 import T5Embedder
from encoders.word2vec import Word2VecEmbedder

def download_spacy_model():
    # Download spaCy English model
    spacy.cli.download("en_core_web_md")

def process_model(model_name, embedder_class, df):
    try:
        embedder = embedder_class()
        df[f'{model_name.lower()}_embeddings'] = df['review'].apply(embedder.get_embeddings)
        df.to_pickle(f'embeddings/yelp_{model_name}_embeddings.pkl')
        print(f"Generated {model_name} embeddings\n{'='*40}\n")
    except Exception as e:
        print(f"Error processing {model_name}: {e}")

def main():
    # Download spaCy model before processing
    download_spacy_model()

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
