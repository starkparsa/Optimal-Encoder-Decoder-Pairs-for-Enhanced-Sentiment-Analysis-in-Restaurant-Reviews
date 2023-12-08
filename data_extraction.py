import pandas as pd
import matplotlib.pyplot as plt

# Specify the path to the Yelp Academic Dataset's review data in JSON format
json_file_path = 'data/yelp_academic_dataset_review.json'

# Create an empty DataFrame to store the chunks
df_list = []

# Read JSON file in chunks and append to the list
chunk_size = 10000
desired_records = 20000
records_read = 0

chunks = pd.read_json(json_file_path, lines=True, chunksize=chunk_size)

for chunk in chunks:
    df_list.append(chunk)
    records_read += chunk.shape[0]

    # Break the loop if the desired number of records is reached
    if records_read >= desired_records:
        break

# Concatenate the list of chunks into a single DataFrame
df = pd.concat(df_list, ignore_index=True)


# Restrict the DataFrame to 'Rating' and 'Review' columns
df = df[['business_id', 'stars', 'text']]

# Rename the 'Rating' column to 'labels'
df = df.rename(columns={'stars': 'labels'})
df = df.rename(columns={'text': 'review'})

# Remove rows where 'Review' is NaN
df = df.dropna(subset=['review'])


import matplotlib.pyplot as plt

# Assuming df is your DataFrame with a 'labels' column
plt.hist(df['labels'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], edgecolor='black', align='mid', rwidth=0.8)  # Adjust the bin edges and other parameters as needed
plt.title('Distribution of Ratings')
plt.xlabel('Ratings')
plt.ylabel('Frequency')
plt.xticks(range(1, 6))  # Adjust the range based on your label values

# Save the histogram as an image file (e.g., PNG)
plt.savefig('results/histogram.png')

plt.show()



# Count the number of samples for each class
class_counts = df['labels'].value_counts()

# Determine the minimum class count
min_class_count = 1000

# Sample an equal number of reviews from each class
balanced_df = df.groupby('labels').apply(lambda x: x.sample(n=min_class_count)).reset_index(drop=True)

# Shuffle the DataFrame
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

# Continue with your data processing steps (e.g., renaming columns, balancing classes, etc.)

# Save the processed data to a pickle file or another format
balanced_df.to_pickle('data/yelp_processed.pkl')

# Display the first few rows of the processed dataset
print(balanced_df.head())

# Number of rows in the processed DataFrame
print("Processed DataFrame rows:", balanced_df.shape[0])