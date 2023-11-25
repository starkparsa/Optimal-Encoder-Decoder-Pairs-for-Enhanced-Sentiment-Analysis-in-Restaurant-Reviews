import pandas as pd
import re
from BERT import BertEmbedder


import pandas as pd

# Replace 'path_to_file' with the actual path to your file
file_path = 'processed_IMDB_Dataset.pkl'

# Read the pickle file into a DataFrame
df = pd.read_pickle(file_path)

# Now, you can work with the DataFrame 'df'
print(df.head())  # Display the first few rows of the DataFrame


# Initialize the BERT embedder
bert_embedder = BertEmbedder()

# Apply the BERT model to each review and create a new column for embeddings
df['bert_embeddings'] = df['review'].apply(bert_embedder.get_embeddings)

# Save the processed data to a pickle file
df.to_pickle('BERT_base_uncased_IMDB_Embeddings.pkl')

# Display the first few rows of the processed dataset
print(df.head())