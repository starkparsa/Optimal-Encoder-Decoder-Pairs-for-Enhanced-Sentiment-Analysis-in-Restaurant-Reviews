import os
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt

def extract_metrics_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        report_dict = pickle.load(file)

    # Extracting metrics values
    metrics_values = {
        'precision': report_dict['precision'],
        'recall': report_dict['recall'],
        'f1-score': report_dict['f1-score'],
        'support': report_dict['support'],
        'accuracy': report_dict['accuracy']
    }

    return metrics_values

def plot_accuracy_table(encoder_name, decoder_name, accuracy_data):
    df = pd.DataFrame(accuracy_data, columns=[f'{decoder_name}'])
    
    # Plotting the table
    ax = pd.plotting.table(data=df, loc='center', colWidths=[0.1] * len(df.columns))
    ax.axis('off')  # Hide the axes

    # Add a title
    ax.set_title(f'Accuracy Table - {encoder_name} + {decoder_name}')

    # Show the table
    plt.show()

def main():
    folder_path = 'results'  # Change this to the folder containing your pickle files

    for file_name in os.listdir(folder_path):
        match = re.match(r'yelp_(BERT|word2vec|BART|T5)_(Random_Forest|SVM|CNN|MLP|logistic_regression|Gradient_Boosting).pkl', file_name)
        if file_name.endswith('.pkl') and match:
            encoder_name, decoder_name = match.groups()
            file_path = os.path.join(folder_path, file_name)

            metrics_values = extract_metrics_from_pickle(file_path)
            accuracy_data = [metrics_values['accuracy']]

            plot_accuracy_table(encoder_name, decoder_name, accuracy_data)

if __name__ == "__main__":
    main()
