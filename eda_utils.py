
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
from scipy.stats import skew
from matplotlib.gridspec import GridSpec

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA

def check_nulls(df):
    a = df.isna().sum()
    for index, value in a.items():
        if value > 0:
            return False
    return True


def smart_normalize(df, skew_threshold=0.5):

    numeric_cols = df.select_dtypes(include=['number']).columns
    df_normalized = df.copy()
    transformations = {}
    
    # Step 1: Skewness correction
    for col in numeric_cols:
        col_skew = skew(df[col].dropna())
        if abs(col_skew) > skew_threshold:
            # Apply Yeo-Johnson power transform (handles positive/negative values)
            pt = PowerTransformer(method='yeo-johnson')
            df_normalized[col] = pt.fit_transform(df[[col]]).flatten()
            transformations[col] = {
                'step1': 'power_transform',
                'skewness': col_skew,
                'transformer': pt
            }
        else:
            transformations[col] = {
                'step1': 'none',
                'skewness': col_skew
            }
    
    # Step 2: Standard scaling (applied to all numeric columns)
    scaler = StandardScaler()
    df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
    transformations['_standard_scaler'] = scaler
    
    return df_normalized, transformations


def remove_outliers(df, outliers_cols, threshold_percent = 1):
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_clean = df.copy()
    
    for i, (col, states) in enumerate(outliers_cols.items()):
        if states['percent'] < threshold_percent:
            lower_bound = states['lower_bound']
            upper_bound = states['upper_bound']
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
    return df_clean


def find_high_outlier_columns(df):
    outlier_columns = {}
    
    for col in df.select_dtypes(include=np.number).columns:
        if col == 'age' or df[col].nunique() < 10:  
            continue
            
        Q1 = df[col].quantile(0.1)
        Q3 = df[col].quantile(0.9)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_percent = (len(outliers) / len(df[col].dropna())) * 100
        
        outlier_columns[col] = {
            'percent': outlier_percent,
            'count': len(outliers),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
            
    return outlier_columns

def visualize_outliers(df, outlier_columns):
    if not outlier_columns:
        print("No columns with more than 5% outliers found.")
        return
        
    for i, (col, stats) in enumerate(outlier_columns.items()):
        if stats['percent'] > 5:
            plt.figure(figsize=(15, 8))
            gs = GridSpec(2, 2, height_ratios=[3, 1])
            
            # Box plot
            ax1 = plt.subplot(gs[0, 0])
            sns.boxplot(x=df[col], ax=ax1)
            ax1.set_title(f"Box Plot: {col}")
            
            # Histogram with KDE
            ax2 = plt.subplot(gs[0, 1])
            sns.histplot(df[col], kde=True, ax=ax2)
            ax2.axvline(stats['lower_bound'], color='r', linestyle='--', label='Outlier Threshold')
            ax2.axvline(stats['upper_bound'], color='r', linestyle='--')
            ax2.legend()
            ax2.set_title(f"Distribution: {col}")
            
            # Outlier detail table
            ax3 = plt.subplot(gs[1, :])
            ax3.axis('off')
            outlier_text = (
                f"Column: {col}\n"
                f"Outliers: {stats['count']} values ({stats['percent']:.2f}% of non-null data)\n"
                f"Lower bound: {stats['lower_bound']:.2f}\n"
                f"Upper bound: {stats['upper_bound']:.2f}\n"
                f"Min: {df[col].min():.2f}, Max: {df[col].max():.2f}\n"
                f"Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}"
            )
            ax3.text(0.5, 0.5, outlier_text, ha='center', va='center', fontsize=12)
            
            plt.tight_layout()
            plt.show()
            
            # Print summary
            print(f"\n📊 Outlier Analysis for {col}:")
            print(f"  • {stats['percent']:.2f}% of values are outliers ({stats['count']} out of {len(df[col].dropna())})")
            print(f"  • Outlier thresholds: < {stats['lower_bound']:.2f} or > {stats['upper_bound']:.2f}")
            print("──"*40)


def view_outlier_dist(df, outlier_columns):
    for i, (col, stats) in enumerate(outlier_columns.items()):
        if stats['percent'] > 5:
            df_clean = df.copy()
            df = df_clean[(df_clean[col] >= stats['lower_bound']) & (df_clean[col] <= stats['upper_bound'])]
            
            male_df = df_clean[df_clean['gender'] == 'male']
            female_df = df_clean[df_clean['gender'] == 'female']
                
            male_stats = {'mean': male_df[col].mean(), 'std': male_df[col].std()}
            female_stats = {'mean': female_df[col].mean(), 'std': female_df[col].std()}
            
            male_snr = abs(male_stats['mean']) / male_stats['std'] if male_stats['std'] != 0 else 0
            female_snr = abs(female_stats['mean']) / female_stats['std'] if female_stats['std'] != 0 else 0
            
            pooled_std = np.sqrt((male_stats['std']**2 + female_stats['std']**2)/2)
            cohens_d = abs(male_stats['mean'] - female_stats['mean']) / pooled_std
            print(cohens_d)
            
            plt.figure(figsize=(12, 6))
            
            # Distribution plot
            sns.kdeplot(data=df_clean, x=col, hue='gender', fill=True, alpha=0.3, 
                       common_norm=False, palette={'male':'blue', 'female':'orange'}, hue_order=('male', 'female'))
            
            plt.axvline(male_stats['mean'], color='blue', linestyle='--', 
                        label=f"Male: μ = {male_stats['mean']:.2f}, σ = {male_stats['std']:.2f}")
            plt.axvline(female_stats['mean'], color='orange', linestyle='--',
                        label=f"Female: μ = {female_stats['mean']:.2f}, σ = {female_stats['std']:.2f}")
            
            plt.axvspan(male_stats['mean'] - male_stats['std'], male_stats['mean'] + male_stats['std'], 
                        color='blue', alpha=0.1)
            plt.axvspan(female_stats['mean'] - female_stats['std'], female_stats['mean'] + female_stats['std'],
                       color='orange', alpha=0.1)
            
            plt.title(f"{col}\nCohen's d = {cohens_d:.2f} (Male SNR: {male_snr:.2f}, Female SNR: {female_snr:.2f})")
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            # Print comprehensive comparison
            print(f"\n📊 {col}")
            print(f"   Male: μ/σ = {male_snr:.2f} (μ = {male_stats['mean']:.2f}, σ = {male_stats['std']:.2f})")
            print(f" Female: μ/σ = {female_snr:.2f} (μ = {female_stats['mean']:.2f}, σ = {female_stats['std']:.2f})")
            print(f" Standardized difference (Cohen's d): {cohens_d:.2f}")
            print("──"*30)


def plot_male_female_diff(male_df, female_df):
    for col in male_df.select_dtypes(include=np.number).columns:  # Only numeric columns
        if col == 'gender':
            continue
            
        male_stats = {'mean': male_df[col].mean(), 'std': male_df[col].std()}
        female_stats = {'mean': female_df[col].mean(), 'std': female_df[col].std()}
        
        male_snr = abs(male_stats['mean']) / male_stats['std'] if male_stats['std'] != 0 else 0
        female_snr = abs(female_stats['mean']) / female_stats['std'] if female_stats['std'] != 0 else 0
        
        pooled_std = np.sqrt((male_stats['std']**2 + female_stats['std']**2)/2)
        cohens_d = abs(male_stats['mean'] - female_stats['mean']) / pooled_std
        
        if cohens_d > 0.4:
            plt.figure(figsize=(12, 6))
            
            # Distribution plot
            sns.kdeplot(data=df, x=col, hue='gender', fill=True, alpha=0.3, 
                       common_norm=False, palette={'male':'blue', 'female':'orange'}, hue_order=('male', 'female'))
            
            plt.axvline(male_stats['mean'], color='blue', linestyle='--', 
                        label=f"Male: μ = {male_stats['mean']:.2f}, σ = {male_stats['std']:.2f}")
            plt.axvline(female_stats['mean'], color='orange', linestyle='--',
                        label=f"Female: μ = {female_stats['mean']:.2f}, σ = {female_stats['std']:.2f}")
            
            plt.axvspan(male_stats['mean'] - male_stats['std'], male_stats['mean'] + male_stats['std'], 
                        color='blue', alpha=0.1)
            plt.axvspan(female_stats['mean'] - female_stats['std'], female_stats['mean'] + female_stats['std'],
                       color='orange', alpha=0.1)
            
            plt.title(f"{col}\nCohen's d = {cohens_d:.2f} (Male SNR: {male_snr:.2f}, Female SNR: {female_snr:.2f})")
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            # Print comprehensive comparison
            print(f"\n📊 {col}")
            print(f"   Male: μ/σ = {male_snr:.2f} (μ = {male_stats['mean']:.2f}, σ = {male_stats['std']:.2f})")
            print(f" Female: μ/σ = {female_snr:.2f} (μ = {female_stats['mean']:.2f}, σ = {female_stats['std']:.2f})")
            print(f" Standardized difference (Cohen's d): {cohens_d:.2f}")
            print("──"*30)
            
def plot_twenties_vs_fifties(twenties_df, fifties_df):
    for col in twenties_df.select_dtypes(include=np.number).columns:  
        if col == 'age':  
            continue
            
        # Calculate statistics
        twenties_stats = {'mean': twenties_df[col].mean(), 'std': twenties_df[col].std()}
        fifties_stats = {'mean': fifties_df[col].mean(), 'std': fifties_df[col].std()}
        
        # Calculate signal-to-noise ratio (mean/std)
        twenties_snr = abs(twenties_stats['mean']) / twenties_stats['std'] if twenties_stats['std'] != 0 else 0
        fifties_snr = abs(fifties_stats['mean']) / fifties_stats['std'] if fifties_stats['std'] != 0 else 0
        
        # Calculate standardized mean difference (Cohen's d)
        pooled_std = np.sqrt((twenties_stats['std']**2 + fifties_stats['std']**2)/2)
        cohens_d = abs(twenties_stats['mean'] - fifties_stats['mean']) / pooled_std
        
        # Only plot if substantial difference exists (Cohen's d > 0.4 effect)
        if cohens_d > 0.2:
            plt.figure(figsize=(12, 6))
            
            # Create temp dataframe with age groups for plotting
            plot_df = pd.concat([
                twenties_df[col].to_frame().assign(age_group='20-29'),
                fifties_df[col].to_frame().assign(age_group='50-59')
            ])
            
            # Distribution plot
            sns.kdeplot(data=plot_df, x=col, hue='age_group', fill=True, alpha=0.3, 
                       common_norm=False, palette={'20-29':'blue', '50-59':'orange'})
            
            # Add statistics annotations
            plt.axvline(twenties_stats['mean'], color='blue', linestyle='--', 
                        label=f"twenties: μ = {twenties_stats['mean']:.2f}, σ = {twenties_stats['std']:.2f}")
            plt.axvline(fifties_stats['mean'], color='orange', linestyle='--',
                        label=f"fifties: μ = {fifties_stats['mean']:.2f}, σ = {fifties_stats['std']:.2f}")
            
            # Add std ranges
            plt.axvspan(twenties_stats['mean'] - twenties_stats['std'], twenties_stats['mean'] + twenties_stats['std'], 
                        color='blue', alpha=0.1)
            plt.axvspan(fifties_stats['mean'] - fifties_stats['std'], fifties_stats['mean'] + fifties_stats['std'],
                       color='orange', alpha=0.1)
            
            plt.title(f"{col}\nCohen's d = {cohens_d:.2f} (twenties SNR: {twenties_snr:.2f}, fifties SNR: {fifties_snr:.2f})")
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            # Print comprehensive comparison
            print(f"\n📊 {col}")
            print(f"   twenties: μ/σ = {twenties_snr:.2f} (μ = {twenties_stats['mean']:.2f}, σ = {twenties_stats['std']:.2f})")
            print(f"   fifties: μ/σ = {fifties_snr:.2f} (μ = {fifties_stats['mean']:.2f}, σ = {fifties_stats['std']:.2f})")
            print(f"   Standardized difference (Cohen's d): {cohens_d:.2f}")
            print("──"*30)


def plot_age_gender_dist(twenties_df, fifties_df):
    for col in twenties_df.select_dtypes(include=np.number).columns:
        if col == 'age':
            continue
            
        # Create subsets for each demographic group
        male_20 = twenties_df[twenties_df['gender'] == 'male'][col].dropna()
        male_50 = fifties_df[fifties_df['gender'] == 'male'][col].dropna()
        female_20 = twenties_df[twenties_df['gender'] == 'female'][col].dropna()
        female_50 = fifties_df[fifties_df['gender'] == 'female'][col].dropna()
        
        # Calculate all Cohen's d comparisons
        def calculate_cohens_d(group1, group2):
            pooled_std = np.sqrt((group1.std()**2 + group2.std()**2)/2)
            return abs(group1.mean() - group2.mean()) / pooled_std
        
        cohens_d = {
            'male_vs_female_20': calculate_cohens_d(male_20, female_20),
            'male_vs_female_50': calculate_cohens_d(male_50, female_50),
            'age_effect_male': calculate_cohens_d(male_20, male_50),
            'age_effect_female': calculate_cohens_d(female_20, female_50)
        }
        
        # Only plot if any comparison shows meaningful effect size
        if any(d > 0.2 for d in cohens_d.values()):
            plt.figure(figsize=(14, 7))
            
            # Create plot dataframe
            plot_df = pd.concat([
                male_20.to_frame().assign(group='Male 20-29'),
                male_50.to_frame().assign(group='Male 50-59'),
                female_20.to_frame().assign(group='Female 20-29'),
                female_50.to_frame().assign(group='Female 50-59')
            ])
            
            # Plot KDE
            palette = {'Male 20-29':'blue', 'Male 50-59':'lightblue',
                     'Female 20-29':'red', 'Female 50-59':'orange'}
            
            for group, color in palette.items():
                group_data = plot_df[plot_df['group'] == group]
                sns.kdeplot(data=group_data, x=col, color=color, 
                            fill=True, alpha=0.2, common_norm=False, 
                            linewidth=2, label=group)
            
            # Add effect size annotations
            text_y = 0.9
            for name, d in cohens_d.items():
                if d > 0.2:
                    comp_name = name.replace('_', ' ').title()
                    plt.text(0.02, text_y, f"{comp_name}: Cohen's d = {d:.2f}",
                            transform=plt.gca().transAxes, fontsize=10,
                            bbox=dict(facecolor='white', alpha=0.7))
                    text_y -= 0.08
            
            plt.title(f"Distribution of {col}\n(Only showing comparisons with Cohen's d > 0.2)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            
            # Print detailed statistics
            print(f"\n📊 {col} - Effect Size Analysis:")
            print(f"  Male 20s vs Female 20s: d = {cohens_d['male_vs_female_20']:.2f}")
            print(f"  Male 50s vs Female 50s: d = {cohens_d['male_vs_female_50']:.2f}")
            print(f"  Age Effect (Male 20s vs 50s): d = {cohens_d['age_effect_male']:.2f}")
            print(f"  Age Effect (Female 20s vs 50s): d = {cohens_d['age_effect_female']:.2f}")
            print("──"*40)


def duration_plot(male_df, female_df):
    # Calculate statistics
    twenties_stats = {'mean': male_df['duration'].mean(), 'std': male_df['duration'].std()}
    fifties_stats = {'mean': female_df['duration'].mean(), 'std': female_df['duration'].std()}
    
    # Calculate signal-to-noise ratio (mean/std)
    twenties_snr = abs(twenties_stats['mean']) / twenties_stats['std'] if twenties_stats['std'] != 0 else 0
    fifties_snr = abs(fifties_stats['mean']) / fifties_stats['std'] if fifties_stats['std'] != 0 else 0
    
    # Calculate standardized mean difference (Cohen's d)
    pooled_std = np.sqrt((twenties_stats['std']**2 + fifties_stats['std']**2)/2)
    cohens_d = abs(twenties_stats['mean'] - fifties_stats['mean']) / pooled_std
    
    plt.figure(figsize=(12, 6))
    
    # Create temp dataframe with age groups for plotting
    plot_df = pd.concat([
        twenties_df['duration'].to_frame().assign(age_group='20-29'),
        fifties_df['duration'].to_frame().assign(age_group='50-59')
    ])
    
    # Distribution plot
    sns.kdeplot(data=plot_df, x='duration', hue='age_group', fill=True, alpha=0.3, 
               common_norm=False, palette={'20-29':'blue', '50-59':'orange'})
    
    # Add statistics annotations
    plt.axvline(twenties_stats['mean'], color='blue', linestyle='--', 
                label=f"twenties: μ = {twenties_stats['mean']:.2f}, σ = {twenties_stats['std']:.2f}")
    plt.axvline(fifties_stats['mean'], color='orange', linestyle='--',
                label=f"fifties: μ = {fifties_stats['mean']:.2f}, σ = {fifties_stats['std']:.2f}")
    
    # Add std ranges
    plt.axvspan(twenties_stats['mean'] - twenties_stats['std'], twenties_stats['mean'] + twenties_stats['std'], 
                color='blue', alpha=0.1)
    plt.axvspan(fifties_stats['mean'] - fifties_stats['std'], fifties_stats['mean'] + fifties_stats['std'],
               color='orange', alpha=0.1)
    
    plt.title(f"{col}\nCohen's d = {cohens_d:.2f} (twenties SNR: {twenties_snr:.2f}, fifties SNR: {fifties_snr:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print comprehensive comparison
    print(f"\n📊 duration")
    print(f"   twenties: μ/σ = {twenties_snr:.2f} (μ = {twenties_stats['mean']:.2f}, σ = {twenties_stats['std']:.2f})")
    print(f"   fifties: μ/σ = {fifties_snr:.2f} (μ = {fifties_stats['mean']:.2f}, σ = {fifties_stats['std']:.2f})")
    print(f"   Standardized difference (Cohen's d): {cohens_d:.2f}")
    print("──"*30)  


def analyze_feature_correlations(df, threshold=0.5):
    
    numeric_df = df.select_dtypes(include=['number'])
    
    corr_matrix = numeric_df.corr()
    
    # Find high correlations
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], 
                                 corr_matrix.iloc[i, j]))
    return high_corr

def features_to_keep(all_features, corr_features, to_be_removed):
    features_set = set()
    to_keep = []
    for removed in to_be_removed:
        features_set.add(removed)
    
    for x, y, z in corr_features:
        if x not in features_set and y not in features_set:
            to_keep.append(x)
            features_set.add(x)
            features_set.add(y)

    for feature in all_features:
        if feature not in features_set:
            to_keep.append(feature)

    return to_keep

def baseline(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return clf.feature_importances_

