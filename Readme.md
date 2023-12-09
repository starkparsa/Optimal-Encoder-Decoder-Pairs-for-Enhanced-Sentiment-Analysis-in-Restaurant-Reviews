# Strategies for Elevated Accuracy:Optimizing Sentiment Analysis with Restaurant Reviews

## Overview
This project focuses on sentiment analysis of Yelp reviews and utilizes various natural language processing techniques to recommend whether a user should visit a restaurant based on the sentiment expressed in the reviews. The sentiment analysis is performed using state-of-the-art embedding models like BERT, BART, and others. The processed data is then used to train and evaluate machine learning classifiers such as Support Vector Machines (SVM) and Logistic Regression.

## Project Structure

### 1. Data Extraction
- The project starts with extracting Yelp review data from the Yelp Academic Dataset in JSON format.
- The extracted data is then processed in chunks, and relevant information such as business ID, rating (stars), and review text is selected.

### 2. Data Exploration and Visualization
- A histogram is created to visualize the distribution of ratings (stars) in the dataset.
- A balanced dataset is created by sampling an equal number of reviews from each class (rating).

### 3. Embeddings Extraction
- Various embedding models, including BERT, BART, and Word2Vec, are used to convert review text into dense vector representations.
- The embeddings are saved as pickle files for future use.

### 4. Model Training and Evaluation
- Machine learning models, including SVM, Logistic Regression, CNN, and others, are trained and evaluated using the processed embeddings.
- Evaluation metrics such as accuracy, precision, recall, and F1-score are saved in CSV files for analysis.

### 5. Results Analysis
- Evaluation metrics are collected and analyzed to compare the performance of different models and embedding techniques.
- The results are visualized using tables and saved as CSV files for further analysis.

### 6. Model Inference
- The trained models are then used to make predictions on a sample restaurant review.
- The recommendations are based on the sentiment expressed in the review.

## Setup and Dependencies
- Install the required dependencies listed in the `requirements.txt` file using the command `pip install -r requirements.txt`.
- Ensure that spaCy models are downloaded using the provided `download_spacy_model` function.

## How to Run
1. Run the main file: "python3 main.py".

## Project Contributors
- Venkata Vaibhav Parasa(vvp23)
- Hasan Angel Bazzi Sabra(hb21h)
- Sai Teja Yapuram Ramesh(sy23f)