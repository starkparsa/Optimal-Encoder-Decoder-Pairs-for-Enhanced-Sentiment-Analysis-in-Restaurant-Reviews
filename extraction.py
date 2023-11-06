import os
import pandas as pd

# Define the directories for positive and negative reviews
positive_data_dir = r'C:\Users\Hasan Angel\CAP_5771_P\aclImdb\train\pos'
negative_data_dir = r'C:\Users\Hasan Angel\CAP_5771_P\aclImdb\train\neg'

# Function to extract text from a file
def extract_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# List to store data
data = []

# Iterate through files in the negative reviews directory
for filename in os.listdir(negative_data_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(negative_data_dir, filename)
        text = extract_text(file_path)
        label = 0  # Negative label
        data.append({'text': text, 'label': label})

# Iterate through files in the positive reviews directory
for filename in os.listdir(positive_data_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(positive_data_dir, filename)
        text = extract_text(file_path)
        label = 1  # Positive label
        data.append({'text': text, 'label': label})

# Create a pandas DataFrame from the data
df = pd.DataFrame(data)

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Save the shuffled dataset to a CSV file
df.to_csv('shuffled_dataset.csv', index=False)

# Display the first few rows of the shuffled dataset
print(df.head())
