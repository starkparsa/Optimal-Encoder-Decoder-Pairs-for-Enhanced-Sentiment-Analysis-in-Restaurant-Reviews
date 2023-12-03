import pandas as pd
from encoders.BART import BartEmbedder
from encoders.BERT import BertEmbedder
from encoders.ERNIE import ErnieEmbedder
from encoders.T5 import T5Embedder
from encoders.word2vec import Word2VecEmbedder
from encoders.XLNet import XLNetEmbedder

file_path = 'data\yelp_processed.pkl'

#####################################################
#                        # BART #

# Read the pickle file into a DataFrame
df = pd.read_pickle(file_path)

# Now, you can work with the DataFrame 'df'
print("Generating BART embeddings")  # Display the first few rows of the DataFrame

# Initialize the BART embedder
bart_embedder = BartEmbedder()

# Apply the BART model to each review and create a new column for embeddings
df['bart_embeddings'] = df['review'].apply(bart_embedder.get_embeddings)

# Save the processed data to a pickle file
df.to_pickle('embeddings\yelp_BART_embeddings.pkl')

# Display the first few rows of the processed dataset
print(df.head())

#####################################################
#                        # BERT #

# Read the pickle file into a DataFrame
df = pd.read_pickle(file_path)

# Now, you can work with the DataFrame 'df'
print("Generating BERT embeddings")  # Display the first few rows of the DataFrame

# Initialize the BERT embedder
bert_embedder = BertEmbedder()

# Apply the BERT model to each review and create a new column for embeddings
df['bert_embeddings'] = df['review'].apply(bert_embedder.get_embeddings)

# Save the processed data to a pickle file
df.to_pickle('embeddings\yelp_BERT_embeddings.pkl')

# Display the first few rows of the processed dataset
print(df.head())

#####################################################
#                        # T5 #

# Now, you can work with the DataFrame 'df'
print("Generating T5 embeddings")  # Display the first few rows of the DataFrame

# Initialize the T5 embedder
t5_embedder = T5Embedder()

# Apply the T5 model to each review and create a new column for embeddings
df['t5_embeddings'] = df['review'].apply(t5_embedder.get_embeddings)

# Save the processed data to a pickle file
df.to_pickle('embeddings\yelp_T5_embeddings.pkl')

# Display the first few rows of the processed dataset
print(df.head())

#####################################################
#                        # ERNIE #

# Now, you can work with the DataFrame 'df'
print("Generating ERNIE embeddings")  # Display the first few rows of the DataFrame

# Initialize the ERNIE embedder
ernie_embedder = ErnieEmbedder()

# Apply the ERNIE model to each review and create a new column for embeddings
df['ernie_embeddings'] = df['review'].apply(ernie_embedder.get_embeddings)

# Save the processed data to a pickle file
df.to_pickle('embeddings\yelp_ERNIE_embeddings.pkl')

# Display the first few rows of the processed dataset
print(df.head())

#####################################################
#                        # XLNet #

# Now, you can work with the DataFrame 'df'
print("Generating XLNet embeddings")  # Display the first few rows of the DataFrame

# Initialize the XLNet embedder
xlnet_embedder = XLNetEmbedder()

# Apply the XLNet model to each review and create a new column for embeddings
df['xlnet_embeddings'] = df['review'].apply(xlnet_embedder.get_embeddings)

# Save the processed data to a pickle file
df.to_pickle('embeddings\yelpt_XLNet_embeddings.pkl')

# Display the first few rows of the processed dataset
print(df.head())