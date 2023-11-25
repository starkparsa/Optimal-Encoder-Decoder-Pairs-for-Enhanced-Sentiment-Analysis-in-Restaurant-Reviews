import pandas as pd
import re
from encoders.BERT import BertEmbedder
from encoders.RoBERTa import RobertaEmbedder
from encoders.BART import BartEmbedder
from encoders.word2vec import Word2VecEmbedder

import pandas as pd




file_path = 'data/processed_IMDB_Dataset.pkl'


#                         # BERT #

# # Read the pickle file into a DataFrame
# df = pd.read_pickle(file_path)

# # Now, you can work with the DataFrame 'df'
# print(df.head())  # Display the first few rows of the DataFrame


# # Initialize the BERT embedder
# bert_embedder = BertEmbedder()

# # Apply the BERT model to each review and create a new column for embeddings
# df['bert_embeddings'] = df['review'].apply(bert_embedder.get_embeddings)

# # Save the processed data to a pickle file
# df.to_pickle('embddings/BERT_base_uncased_IMDB_Embeddings.pkl')

# # Display the first few rows of the processed dataset
# print(df.head())






#                         # RoBERTa #


# # Read the pickle file into a DataFrame
# df = pd.read_pickle(file_path)

# # Now, you can work with the DataFrame 'df'
# print(df.head())  # Display the first few rows of the DataFrame


# # Initialize the BERT embedder
# roberta_embedder = RobertaEmbedder()

# # Apply the BERT model to each review and create a new column for embeddings
# df['bert_embeddings'] = df['review'].apply(roberta_embedder.get_embeddings)

# # Save the processed data to a pickle file
# df.to_pickle('embddings/RoBERTa_base_uncased_IMDB_Embeddings.pkl')

# # Display the first few rows of the processed dataset
# print(df.head())









#                         # BART #

# # Read the pickle file into a DataFrame
# df = pd.read_pickle(file_path)

# # Now, you can work with the DataFrame 'df'
# print(df.head())  # Display the first few rows of the DataFrame


# # Initialize the BERT embedder
# bart_embedder = BartEmbedder()

# # Apply the BERT model to each review and create a new column for embeddings
# df['bart_embeddings'] = df['review'].apply(bart_embedder.get_embeddings)

# # Save the processed data to a pickle file
# df.to_pickle('embddings/BART_base_uncased_IMDB_Embeddings.pkl')

# # Display the first few rows of the processed dataset
# print(df.head())





                        #  Word2Vec #

# Read the pickle file into a DataFrame
df = pd.read_pickle(file_path)

# Now, you can work with the DataFrame 'df'
print(df.head())  # Display the first few rows of the DataFrame


# Initialize the BERT embedder
Word2Vec_embedder = Word2VecEmbedder()

# Apply the BERT model to each review and create a new column for embeddings
df['Word2Vec_embeddings'] = df['review'].apply(Word2Vec_embedder.get_embeddings)

# Save the processed data to a pickle file
df.to_pickle('embddings/Word2Ve_base_IMDB_Embeddings.pkl')

# Display the first few rows of the processed dataset
print(df.head())
