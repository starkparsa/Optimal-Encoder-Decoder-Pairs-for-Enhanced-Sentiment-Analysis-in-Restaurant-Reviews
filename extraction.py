import pandas as pd

# Load the dataset
df = pd.read_csv('data\marketing_sample_for_walmart_com-walmart_product_reviews__20200401_20200630__30k_data.csv')

# Restrict the DataFrame to 'Rating' and 'Review' columns
df = df[['Rating', 'Review']]

# Rename the 'Rating' column to 'labels'
df = df.rename(columns={'Rating': 'labels'})
df = df.rename(columns={'Review': 'review'})

# Remove rows where 'Review' is NaN
df = df.dropna(subset=['review'])

# Count the number of samples for each class
class_counts = df['labels'].value_counts()

# Determine the minimum class count
min_class_count = class_counts.min()

# Sample an equal number of reviews from each class
balanced_df = df.groupby('labels').apply(lambda x: x.sample(n=min_class_count)).reset_index(drop=True)

# Shuffle the DataFrame
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

# Save the processed data to a pickle file
balanced_df.to_pickle('data/walmart_processed.pkl')

# Display the first few rows of the balanced dataset
print(balanced_df.head())


# Number of rows in the original DataFrame
print("Original DataFrame rows:", df.shape[0])

# Number of rows in the balanced DataFrame
print("Balanced DataFrame rows:", balanced_df.shape[0])