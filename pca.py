import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def read_csv_in_chunks(file_path, chunk_size=10000):
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

def perform_pca(input_file: str, output_file: str, n_components: int = 2):
    print("Reading file in chunks...")
    df = read_csv_in_chunks(input_file)

    # Drop non-numeric or target columns like 'gender', 'age' if needed
    features = df.select_dtypes(include=['float64', 'int64']).drop(columns=['gender', 'age'], errors='ignore')

    print("Performing PCA...")
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features)

    pca_df = pd.DataFrame(data=principal_components, columns=[f"PC{i+1}" for i in range(n_components)])

    # Optionally add 'gender' or 'age' back for labeling/visualization
    if 'gender' in df.columns:
        pca_df['gender'] = df['gender']
    if 'age' in df.columns:
        pca_df['age'] = df['age']

    print("Saving results...")
    pca_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    perform_pca(
        input_file="normalized_extracted_features.csv",
        output_file="pca_result.csv",
        n_components=10
    )
