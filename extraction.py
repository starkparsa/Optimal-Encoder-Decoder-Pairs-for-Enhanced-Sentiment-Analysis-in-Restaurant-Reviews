import pandas as pd
import re

# Load the dataset
df = pd.read_csv('data/IMDB_Dataset.csv')

# Mapping labels to numerical values
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Remove <br /><br /> from reviews using regular expression
df['review'] = df['review'].apply(lambda x: re.sub(r'<br /><br />', '', x))

# Shuffle the DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# Save the processed data to a pickle file
df.to_pickle('data/processed_IMDB_Dataset.pkl')

# Display the first few rows of the processed dataset
print(df.head())
