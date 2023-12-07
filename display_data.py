import pandas as pd
from pathlib import Path
import pickle

# Sample data for demonstration purposes (replace this with your actual data)
file_format = 'yelp_{}_{}.pkl'
encoders = ['Word2Vec', 'BERT', 'BART', 'T5']
decoders = ['logistic_regression', 'svm', 'mlp', 'cnn', 'gradient_boosting', 'random_forest']

# Create a DataFrame to store accuracy values
accuracy_data = pd.DataFrame(index=encoders, columns=decoders)

for encoder in encoders:
    for decoder in decoders:
        file_path = Path('results') / file_format.format(encoder, decoder)
        
        try:
            with open(file_path, 'rb') as file:
                results_data = pickle.load(file)

            # Assuming 'accuracy' is a key within the dictionary
            accuracy_value = results_data.get('accuracy', None)

            accuracy_data.loc[encoder, decoder] = accuracy_value
        except FileNotFoundError:
            print(f"File not found: {file_path}")

# Display the table
print("\nAccuracy Table:\n")
print(accuracy_data)

# Save the DataFrame to a CSV file
csv_filename = 'results/accuracy_table.csv'
accuracy_data.to_csv(csv_filename)
print(f"Accuracy table saved to {csv_filename}")