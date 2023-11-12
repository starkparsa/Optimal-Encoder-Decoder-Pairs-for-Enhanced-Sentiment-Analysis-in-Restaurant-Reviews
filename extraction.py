import pandas as pd
import re

# Load the dataset
df = pd.read_csv('IMDB_Dataset.csv')

# Mapping labels to numerical values
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Remove <br /><br /> from reviews using regular expression
df['review'] = df['review'].apply(lambda x: re.sub(r'<br /><br />', '', x))

# Save the processed data to a new CSV file
df.to_csv('processed_IMDB_Dataset.csv', index=False)

# Display the first few rows of the processed dataset
print(df.head())
