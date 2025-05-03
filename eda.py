import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
from scipy.stats import skew
from matplotlib.gridspec import GridSpec
import json
def clean_df(dataframe):
    df = dataframe
    #replace null with mean
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].isnull().sum() > 0:
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)
    file = 'Inference/remaining_features.json'
    with open(file, 'r') as f:
        features = json.load(f)
    df = df[features]
    return df
# x = clean_df(pd.read_csv("denoised.csv"))
# print(x.shape)