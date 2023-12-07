import pandas as pd
from pathlib import Path
import pickle

# Sample data for demonstration purposes (replace this with your actual data)
file_format = 'yelp_{}_{}.pkl'
encoders = ['Word2Vec', 'BERT', 'BART', 'T5']
decoders = ['logistic_regression', 'svm', 'mlp', 'cnn', 'gradient_boosting', 'random_forest']

# Create DataFrames to store evaluation metrics
accuracy_data = pd.DataFrame(index=encoders, columns=decoders)
f1_data = pd.DataFrame(index=encoders, columns=decoders)
precision_data = pd.DataFrame(index=encoders, columns=decoders)
recall_data = pd.DataFrame(index=encoders, columns=decoders)
specificity_data = pd.DataFrame(index=encoders, columns=decoders)

for encoder in encoders:
    for decoder in decoders:
        file_path = Path('results') / file_format.format(encoder, decoder)
        
        try:
            with open(file_path, 'rb') as file:
                results_data = pickle.load(file)

            # Extract evaluation metrics from the results_data dictionary
            accuracy_value = results_data.get('accuracy', None)
            f1_value = results_data.get('weighted avg', {}).get('f1-score', None)
            precision_value = results_data.get('weighted avg', {}).get('precision', None)
            recall_value = results_data.get('weighted avg', {}).get('recall', None)



            # Update the DataFrames
            accuracy_data.loc[encoder, decoder] = accuracy_value
            f1_data.loc[encoder, decoder] = f1_value
            precision_data.loc[encoder, decoder] = precision_value
            recall_data.loc[encoder, decoder] = recall_value

        except FileNotFoundError:
            print(f"File not found: {file_path}")

# Display and save the tables
print("\nAccuracy Table:\n")
print(accuracy_data)
accuracy_csv_filename = 'results/accuracy_table.csv'
accuracy_data.to_csv(accuracy_csv_filename)
print(f"Accuracy table saved to {accuracy_csv_filename}")

print("\nF1 Table:\n")
print(f1_data)
f1_csv_filename = 'results/f1_table.csv'
f1_data.to_csv(f1_csv_filename)
print(f"F1 table saved to {f1_csv_filename}")

print("\nPrecision Table:\n")
print(precision_data)
precision_csv_filename = 'results/precision_table.csv'
precision_data.to_csv(precision_csv_filename)
print(f"Precision table saved to {precision_csv_filename}")

print("\nRecall Table:\n")
print(recall_data)
recall_csv_filename = 'results/recall_table.csv'
recall_data.to_csv(recall_csv_filename)
print(f"Recall table saved to {recall_csv_filename}")
