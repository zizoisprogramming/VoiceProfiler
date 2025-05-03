import os
import pickle
import pandas as pd
import numpy as np
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from preprocessing import *
from cleaning import clean_df
def normalize_test(X_test, transformations):
    X_test_normalized = X_test.copy()
    for col, info in transformations.items():
        if col == '_standard_scaler':
            continue
        if info['step1'] == 'power_transform':
            pt = info['transformer']
            if col in X_test.columns:
                X_test_normalized[col] = pt.transform(X_test[[col]]).flatten()
            else:
                print(f"Column {col} not found in X_test. Skipping power transformation.")

    # Apply standard scaling using the saved scaler
    scaler = transformations['_standard_scaler']
    numeric_cols = X_test.select_dtypes(include=['number']).columns
    X_test_normalized[numeric_cols] = scaler.transform(X_test_normalized[numeric_cols])
    return X_test_normalized
def extract_features():
    files = os.listdir('data/')
    files = sorted(files, key=lambda x: int(x.split('.')[0]))
    features = []
    for file in files:
        feature = extract_features_per_audio('data/' + file)
        features.append(feature)
    all_features = pd.concat(features, ignore_index=True)
    return all_features
def predict(file_path, model_path, output_path, labels_path=None):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    data = pd.read_csv(file_path)
    X = data

    y_pred = model.predict(X)

    # Save predictions to a file
    with open(output_path, 'w') as f:
        for pred in y_pred:
            f.write(f"{pred}\n")
    if labels_path:
        labels = pd.read_csv(labels_path)
        y = labels.iloc[:, 0].values
        accuracy = (y_pred == y).mean()
        print(f"Accuracy: {accuracy:.4f}")
    print("Predictions saved to", output_path)
    print("Current working dir", os.getcwd())

def transform_features(df):
    with open('normalization_transformations.pkl', 'rb') as f:
        transformations = pickle.load(f)
    return normalize_test(df, transformations)
def pipeline(test_file = None, results_file = None):
    if(test_file is None):
        df = extract_features()
    else:
        df = pd.read_csv(test_file)
        df['gender'] = df['gender'].map({'male': 0, 'female': 1, 0: 0, 1: 1})
        df['age'] = df['age'].map({'twenties': 0, 'fifties': 1, 0: 0, 1: 1})
        y = df['gender'] + 2 * df['age']
        df = df.drop(columns = ['age', 'gender'])
        y.to_csv('results.csv', index=False)
    df = clean_df(df)   ## select features
    df = transform_features(df) ## transformations
    df.to_csv('test_data.csv', index=False)

    predict("test_data.csv", "91.pkl", "results.txt", results_file)

if __name__ == "__main__":
    pipeline()
    #pipeline('testing.csv', 'results.csv')