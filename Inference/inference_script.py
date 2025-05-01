import os
import pickle
import pandas as pd
import numpy as np
import os
def feature_extraction():
    return
def eda():
    return
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


predict("test_data.csv", "svc_both.pkl", "predictions.txt", "test_labels.csv")