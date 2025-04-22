import pandas as pd

def normalize_column(column: pd.Series) -> pd.Series:
    mean = column.mean()
    std = column.std()
    
    if std == 0:
        return column - mean
    
    return (column - mean) / std

def main():
    # Path to your CSV file
    input_file = 'extracted_features.csv'
    output_file = 'normalized_extracted_features.csv'

    # Load the CSV
    df = pd.read_csv(input_file)

    # List of column names to normalize
    columns_to_normalize = [
        'f0_mean', 'f0_std', 'formant1', 'formant2', 'mfcc_0_mean',
        'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean',
        'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_7_mean', 'mfcc_8_mean'
    ]

    # Normalize each column
    for col in columns_to_normalize:
        if col in df.columns:
            df[col] = normalize_column(df[col])
        else:
            print(f"Warning: Column '{col}' not found in the CSV.")

    # Save the normalized DataFrame
    df.to_csv(output_file, index=False)
    print(f"Normalized data saved to '{output_file}'.")

if __name__ == '__main__':
    main()
