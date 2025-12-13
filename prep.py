import pandas as pd
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# File paths
DATA_DIR = Path('data')
CSV_FILE = DATA_DIR / 'properties.csv'
DESCRIPTIONS_DIR = DATA_DIR / 'descriptions' / 'descriptions'
IMAGES_DIR = DATA_DIR / 'images' / 'images' # unused
EXTERNAL_IMAGES_DIR = Path(r'C:\Users\gvida\Downloads\images\images')

# Column names based on readme.txt
COLUMNS = [
    'reference',
    'location',
    'price',
    'title',
    'bedrooms',
    'bathrooms',
    'indoor_sqm',
    'outdoor_sqm',
    'features'
]

def load_properties():
    df = pd.read_csv(CSV_FILE, names=COLUMNS, encoding='utf-8')
    print(f"Loaded {len(df)} properties")
    return df

def clean_price(price_str):
    if pd.isna(price_str):
        return None
    # Remove currency symbol and commas
    price_clean = str(price_str).replace('€', '').replace(',', '').strip()
    try:
        return float(price_clean)
    except ValueError:
        return None

def parse_features(features_str):
    if pd.isna(features_str) or features_str == '':
        return []
    return [f.strip() for f in str(features_str).split('|')]

def process_properties(df):
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Track removed properties
    removed_references = []
    
    # Clean price column
    df_processed['price'] = df_processed['price'].apply(clean_price)
    
    # Parse features into lists
    df_processed['features_list'] = df_processed['features'].apply(parse_features)
    
    # Convert numeric columns (handle 'NA' strings)
    numeric_cols = ['bedrooms', 'bathrooms', 'indoor_sqm', 'outdoor_sqm']
    for col in numeric_cols:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Print statistics BEFORE cleaning
    print("\n" + "="*60)
    print("BEFORE DATA CLEANING")
    print("="*60)
    print(f"Total properties: {len(df_processed)}")
    print("\nNumeric variables:")
    for col in ['price', 'bedrooms', 'bathrooms', 'indoor_sqm', 'outdoor_sqm']:
        missing = df_processed[col].isna().sum()
        print(f"  {col:15s}: min={df_processed[col].min():.0f}, max={df_processed[col].max():.0f}, missing={missing} ({missing/len(df_processed)*100:.2f}%)")
    
    # Clean unrealistic values in bedrooms and bathrooms
    # Bedrooms: reasonable range 0-20 (anything above is likely data entry error)
    # Bathrooms: reasonable range 0-15 (anything above is likely data entry error)
    print("="*60)
    print("CLEANING DATA QUALITY ISSUES")
    print("="*60)
    
    bedrooms_outliers = df_processed[df_processed['bedrooms'] > 20]
    if len(bedrooms_outliers) > 0:
        print(f"Found {len(bedrooms_outliers)} properties with bedrooms > 20 (setting to NaN)")
        df_processed.loc[df_processed['bedrooms'] > 20, 'bedrooms'] = np.nan
    
    bathrooms_outliers = df_processed[df_processed['bathrooms'] > 15]
    if len(bathrooms_outliers) > 0:
        print(f"Found {len(bathrooms_outliers)} properties with bathrooms > 15 (setting to NaN)")
        df_processed.loc[df_processed['bathrooms'] > 15, 'bathrooms'] = np.nan
    
    # Drop outdoor_sqm column (70% missing values)
    print(f"Dropping outdoor_sqm column (68.38% missing values)")
    df_processed = df_processed.drop(columns=['outdoor_sqm'])
    
    # Remove rows with missing values first, before handling outliers
    print("\nRemoving rows with missing numeric values:")
    initial_count = len(df_processed)
    
    numeric_cols = ['price', 'bedrooms', 'bathrooms', 'indoor_sqm']
    for col in numeric_cols:
        missing_count = df_processed[col].isna().sum()
        print(f"  {col:15s}: {missing_count} missing values ({missing_count/initial_count*100:.2f}%)")
    
    # Track which references are being removed
    rows_to_remove = df_processed[df_processed[numeric_cols].isna().any(axis=1)]
    removed_references.extend(rows_to_remove['reference'].tolist())
    
    df_processed = df_processed.dropna(subset=numeric_cols)
    rows_deleted = initial_count - len(df_processed)
    print(f"\nTotal rows deleted: {rows_deleted} ({rows_deleted/initial_count*100:.2f}%)")
    print(f"Remaining rows: {len(df_processed)}")
    
    # Handle outliers using IQR method (for all except price)
    print("\n" + "="*60)
    print("HANDLING OUTLIERS (IQR method, multiplier=2.5)")
    print("="*60)
    
    outlier_cols = ['bedrooms', 'bathrooms', 'indoor_sqm']
    IQR_MULTIPLIER = 2.5  # Relaxed from 1.5 to preserve correlations
    for col in outlier_cols:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - IQR_MULTIPLIER * IQR
        upper_bound = Q3 + IQR_MULTIPLIER * IQR
        
        outliers_low = df_processed[df_processed[col] < lower_bound]
        outliers_high = df_processed[df_processed[col] > upper_bound]
        total_outliers = len(outliers_low) + len(outliers_high)
        
        if total_outliers > 0:
            print(f"\n{col}:")
            print(f"  IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"  Outliers: {total_outliers} ({total_outliers/len(df_processed)*100:.2f}%)")
            if len(outliers_low) > 0:
                print(f"    Below lower bound: {len(outliers_low)} (min={outliers_low[col].min():.2f})")
            if len(outliers_high) > 0:
                print(f"    Above upper bound: {len(outliers_high)} (max={outliers_high[col].max():.2f})")
            
            # Cap outliers at bounds
            df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
            df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
            print(f"  Capped to bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # Remove rows with missing values in key numeric columns
    print("\nRemoving rows with missing numeric values:")
    initial_count = len(df_processed)
    
    numeric_cols = ['price', 'bedrooms', 'bathrooms', 'indoor_sqm']
    for col in numeric_cols:
        missing_count = df_processed[col].isna().sum()
        print(f"  {col:15s}: {missing_count} missing values ({missing_count/initial_count*100:.2f}%)")
    
    df_processed = df_processed.dropna(subset=numeric_cols)
    rows_deleted = initial_count - len(df_processed)
    print(f"\nTotal rows deleted: {rows_deleted} ({rows_deleted/initial_count*100:.2f}%)")
    print(f"Remaining rows: {len(df_processed)}")
    
    # Print statistics AFTER cleaning
    print("\n" + "="*60)
    print("AFTER DATA CLEANING")
    print("="*60)
    print(f"Total properties: {len(df_processed)}")
    print("\nNumeric variables:")
    for col in ['price', 'bedrooms', 'bathrooms', 'indoor_sqm']:
        missing = df_processed[col].isna().sum()
        print(f"  {col:15s}: min={df_processed[col].min():.0f}, max={df_processed[col].max():.0f}, missing={missing} ({missing/len(df_processed)*100:.2f}%)")
    
    # Extract property type from title
    df_processed['property_type'] = df_processed['title'].str.extract(r'(\d+\s+Bedroom\s+\w+|\w+\s+Villa|\w+|Plot)', expand=False)
    
    return df_processed, removed_references

def analyze_features(df):
    print("\n" + "="*60)
    print("FEATURE ANALYSIS")
    print("="*60)
    
    # Collect all features across all properties
    all_features = []
    for features_list in df['features_list']:
        all_features.extend(features_list)
    
    # Count occurrences of each feature
    feature_counts = pd.Series(all_features).value_counts()
    
    print(f"\nTotal unique features: {len(feature_counts)}")
    print(f"Total feature instances: {len(all_features)}")
    
    print("\n" + "-"*60)
    print(f"{'Feature':<50} {'Count':>8} {'%':>6}")
    print("-"*60)
    
    for feature, count in feature_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{feature:<50} {count:>8} {percentage:>6.2f}%")
    
    return feature_counts

def create_feature_columns(df, min_count=10, top_n=None, by_correlation=False):
    print("\n" + "="*60)
    if top_n and by_correlation:
        print(f"CREATING FEATURE COLUMNS (top {top_n} by correlation with price)")
    elif top_n:
        print(f"CREATING FEATURE COLUMNS (top {top_n} most common features)")
    else:
        print(f"CREATING FEATURE COLUMNS (min_count={min_count})")
    print("="*60)
    
    # Get feature frequencies
    all_features = []
    for features_list in df['features_list']:
        all_features.extend(features_list)
    
    feature_counts = pd.Series(all_features).value_counts()
    
    # If selecting by correlation, calculate correlations first
    if by_correlation and top_n:
        print("\nCalculating correlations with price for all features...")
        
        # Create temporary binary columns for all features with min_count occurrences
        temp_features = feature_counts[feature_counts >= min_count].index.tolist()
        
        feature_correlations = {}
        for feature in temp_features:
            # Create binary column
            has_feature = df['features_list'].apply(lambda x: feature in x)
            # Calculate correlation with price
            corr = has_feature.corr(df['price'])
            if not np.isnan(corr):
                feature_correlations[feature] = abs(corr)  # Use absolute value
        
        # Sort by correlation and get top N
        sorted_features = sorted(feature_correlations.items(), key=lambda x: x[1], reverse=True)
        valid_features = [f[0] for f in sorted_features[:top_n]]
        
        print(f"\nTop {top_n} features by correlation with price:")
        print("-" * 60)
        for feature, corr in sorted_features[:top_n]:
            actual_corr = df['features_list'].apply(lambda x: feature in x).corr(df['price'])
            count = feature_counts[feature]
            print(f"{feature:<45} {actual_corr:>8.4f} ({count:>5} properties, {count/len(df)*100:>5.1f}%)")
        print(f"\nFeatures excluded: {len(feature_counts) - top_n}")
        
    elif top_n:
        # Select by frequency
        valid_features = feature_counts.head(top_n).index.tolist()
        print(f"\nKeeping top {top_n} most common features")
        print(f"Features excluded: {len(feature_counts) - top_n}")
    else:
        # Select by min_count
        valid_features = feature_counts[feature_counts >= min_count].index.tolist()
        print(f"\nFeatures with at least {min_count} occurrences: {len(valid_features)}")
        print(f"Features excluded (too rare): {len(feature_counts) - len(valid_features)}")
    
    # Create binary columns for each valid feature
    # Use a more efficient approach to avoid fragmentation warnings
    feature_dfs = []
    for feature in valid_features:
        # Clean feature name to make it a valid column name
        col_name = f"has_{feature.replace(' ', '_').replace('/', '_').replace('-', '_')}"
        feature_df = df['features_list'].apply(lambda x: feature in x).to_frame(col_name)
        feature_dfs.append(feature_df)
    
    # Concatenate all feature columns at once
    if feature_dfs:
        features_combined = pd.concat(feature_dfs, axis=1)
        df_with_features = pd.concat([df, features_combined], axis=1)
    else:
        df_with_features = df.copy()
    
    print(f"\nCreated {len(valid_features)} binary feature columns")
    
    # Show sample of created columns
    feature_cols = [col for col in df_with_features.columns if col.startswith('has_')]
    print(f"\nFeature columns: {feature_cols[:5]}...")
    
    return df_with_features, valid_features

def get_description(reference):
    desc_file = DESCRIPTIONS_DIR / f"{reference}.txt"
    if desc_file.exists():
        with open(desc_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None

def get_images(reference):
    images_folder = IMAGES_DIR / reference
    if images_folder.exists() and images_folder.is_dir():
        return list(images_folder.glob('*'))
    return []

def detect_image_content(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        pixels = np.array(img)
        
        # Extract color channels
        red_channel = pixels[:, :, 0]
        green_channel = pixels[:, :, 1]
        blue_channel = pixels[:, :, 2]
        
        # Green detection for gardens/outdoor spaces
        green_dominant = (green_channel > red_channel) & (green_channel > blue_channel)
        green_ratio = green_dominant.sum() / green_channel.size
        
        return {
            'has_greenery': green_ratio > 0.2
        }
    except Exception as e:
        return {
            'has_greenery': False
        }

def extract_image_features(reference, images_dir):
    img_folder = images_dir / reference
    
    features = {
        'image_count': 0,
        'greenery_photo_count': 0
    }
    
    if not img_folder.exists():
        return features
    
    # Get all image files
    image_files = list(img_folder.glob('*.jpg')) + list(img_folder.glob('*.jpeg')) + list(img_folder.glob('*.png'))
    
    features['image_count'] = len(image_files)
    
    if len(image_files) == 0:
        return features
    
    # Analyze images for greenery
    for img_file in image_files:
        content = detect_image_content(img_file)
        
        if content['has_greenery']:
            features['greenery_photo_count'] += 1
    
    return features

def encode_categorical_features(df):
    print("\n" + "="*60)
    print("ENCODING CATEGORICAL FEATURES")
    print("="*60)
    
    # Target encoding: encode by average price per category
    print("\n1. TARGET ENCODING")
    print("-" * 40)
    
    # Location encoding
    location_encoding = df.groupby('location')['price'].mean()
    df['location_encoded'] = df['location'].map(location_encoding)
    location_corr = df['location_encoded'].corr(df['price'])
    print(f"Location encoded: {df['location'].nunique()} unique values")
    print(f"  Correlation with price: {location_corr:.4f}")
    
    # Property type encoding
    property_type_encoding = df.groupby('property_type')['price'].mean()
    df['property_type_encoded'] = df['property_type'].map(property_type_encoding)
    property_type_corr = df['property_type_encoded'].corr(df['price'])
    print(f"\nProperty type encoded: {df['property_type'].nunique()} unique values")
    print(f"  Correlation with price: {property_type_corr:.4f}")
    
    print("\n" + "="*60)
    print(f"Categorical features encoded successfully")
    print(f"Added columns: location_encoded, property_type_encoded")
    print("="*60)
    
    return df

def add_external_data(df):
    df['has_description'] = df['reference'].apply(
        lambda ref: (DESCRIPTIONS_DIR / f"{ref}.txt").exists()
    )
    
    # Initialize description feature columns
    df['desc_length'] = 0
    df['desc_has_luxury'] = False
    df['desc_has_exclusive'] = False
    df['desc_has_stunning'] = False
    
    print("\nExtracting description features...")
    # Extract features from description files
    for idx, row in df.iterrows():
        ref = row['reference']
        desc_file = DESCRIPTIONS_DIR / f"{ref}.txt"
        
        if desc_file.exists():
            try:
                with open(desc_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    content_lower = content.lower()
                
                df.at[idx, 'desc_length'] = len(content)
                df.at[idx, 'desc_has_luxury'] = True if 'luxury' in content_lower or 'luxurious' in content_lower else False
                df.at[idx, 'desc_has_exclusive'] = True if 'exclusive' in content_lower else False
                df.at[idx, 'desc_has_stunning'] = True if 'stunning' in content_lower else False
                
            except Exception as e:
                pass
    
    # Print summary
    print(f"  desc_length: avg={df['desc_length'].mean():.0f} chars")
    print(f"  desc_has_luxury: {df['desc_has_luxury'].sum()} properties ({df['desc_has_luxury'].sum()/len(df)*100:.1f}%)")
    print(f"  desc_has_exclusive: {df['desc_has_exclusive'].sum()} properties ({df['desc_has_exclusive'].sum()/len(df)*100:.1f}%)")
    print(f"  desc_has_stunning: {df['desc_has_stunning'].sum()} properties ({df['desc_has_stunning'].sum()/len(df)*100:.1f}%)")
    
    # Initialize image feature columns
    df['image_count'] = 0
    df['greenery_photo_count'] = 0
    
    # Check if external images directory exists
    if EXTERNAL_IMAGES_DIR.exists():
        print("\nExtracting image features...")
        print(f"  Processing {len(df)} properties...")
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"  Processed {idx}/{len(df)} properties...")
            
            ref = row['reference']
            img_features = extract_image_features(ref, EXTERNAL_IMAGES_DIR)
            
            df.at[idx, 'image_count'] = img_features['image_count']
            df.at[idx, 'greenery_photo_count'] = img_features['greenery_photo_count']
        
        # Print summary
        print(f"\n  Image feature summary:")
        print(f"  image_count: avg={df['image_count'].mean():.1f}, max={df['image_count'].max():.0f}")
        print(f"  greenery_photo_count: avg={df['greenery_photo_count'].mean():.1f}")
        print(f"  properties with greenery photos: {(df['greenery_photo_count'] > 0).sum()} ({(df['greenery_photo_count'] > 0).sum()/len(df)*100:.1f}%)")
    else:
        print(f"\n  Warning: External images directory not found: {EXTERNAL_IMAGES_DIR}")
        print(f"  Skipping image feature extraction.")
    
    return df

def get_summary_stats(df):
    stats = {
        'total_properties': len(df),
        'avg_price': df['price'].mean(),
        'median_price': df['price'].median(),
        'min_price': df['price'].min(),
        'max_price': df['price'].max(),
        'avg_bedrooms': df['bedrooms'].mean(),
        'avg_bathrooms': df['bathrooms'].mean(),
        'avg_indoor_sqm': df['indoor_sqm'].mean(),
        'properties_with_description': df['has_description'].sum()
    }
    return stats

def generate_variable_report(df):
    print("\n" + "="*60)
    print("DETAILED VARIABLE ANALYSIS")
    print("="*60)
    
    numeric_vars = ['price', 'bedrooms', 'bathrooms', 'indoor_sqm']
    
    report_data = []
    for var in numeric_vars:
        stats = {
            'Variable': var,
            'Count': df[var].count(),
            'Missing': df[var].isna().sum(),
            'Missing %': f"{(df[var].isna().sum() / len(df) * 100):.2f}%",
            'Mean': df[var].mean(),
            'Median': df[var].median(),
            'Std Dev': df[var].std(),
            'Min': df[var].min(),
            'Max': df[var].max(),
            'Q25': df[var].quantile(0.25),
            'Q75': df[var].quantile(0.75)
        }
        report_data.append(stats)
    
    report_df = pd.DataFrame(report_data)
    print("\n" + report_df.to_string(index=False))
    
    return report_df

def plot_variable_distributions(df):
    numeric_vars = ['price', 'bedrooms', 'bathrooms', 'indoor_sqm']
    
    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 24))
    
    for idx, var in enumerate(numeric_vars):
        # Clean data (remove NaN for plotting)
        data = df[var].dropna()
        
        # Row position for this variable (3 plots per variable)
        row = idx * 3
        
        # 1. Histogram with KDE
        ax1 = plt.subplot(len(numeric_vars), 3, idx * 3 + 1)
        ax1.hist(data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel(var, fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title(f'{var} - Histogram', fontsize=11, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {data.mean():.2f}\nMedian: {data.median():.2f}\nStd: {data.std():.2f}'
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8)
        
        # 2. Distribution plot with KDE
        ax2 = plt.subplot(len(numeric_vars), 3, idx * 3 + 2)
        data.hist(bins=50, alpha=0.5, ax=ax2, color='steelblue', edgecolor='black', density=True)
        data.plot(kind='kde', ax=ax2, color='red', linewidth=2)
        ax2.set_xlabel(var, fontsize=10)
        ax2.set_ylabel('Density', fontsize=10)
        ax2.set_title(f'{var} - Distribution with KDE', fontsize=11, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Boxplot
        ax3 = plt.subplot(len(numeric_vars), 3, idx * 3 + 3)
        bp = ax3.boxplot(data, vert=True, patch_artist=True, 
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5))
        ax3.set_ylabel(var, fontsize=10)
        ax3.set_title(f'{var} - Boxplot', fontsize=11, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add outlier info
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
        outlier_text = f'Outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)'
        ax3.text(0.5, 0.02, outlier_text, transform=ax3.transAxes,
                horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = DATA_DIR / 'variable_distributions.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nVariable distribution plots saved to: {plot_file}")
    
    # Display the plot
    plt.show()

def analyze_correlations(df):
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS WITH PRICE")
    print("="*60)
    
    # Select numeric columns for correlation
    numeric_vars = ['bedrooms', 'bathrooms', 'indoor_sqm']
    
    # Create correlation matrix with price
    correlations = {}
    for var in numeric_vars:
        # Calculate correlation, ignoring NaN values
        corr = df[[var, 'price']].corr().iloc[0, 1]
        correlations[var] = corr
    
    # Sort by absolute correlation strength
    sorted_correlations = sorted(correlations.items(), 
                                 key=lambda x: abs(x[1]), 
                                 reverse=True)
    
    print("\nCorrelations with Price (sorted by strength):")
    print("-" * 60)
    for var, corr in sorted_correlations:
        if not np.isnan(corr):
            print(f"{var:20s}: {corr:>8.4f}")
    
    # Full correlation matrix for all numeric variables
    print("\n" + "="*60)
    print("FULL CORRELATION MATRIX")
    print("="*60)
    
    corr_vars = ['price'] + numeric_vars
    correlation_matrix = df[corr_vars].corr()
    
    print("\n" + correlation_matrix.to_string())
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bar chart of correlations with price
    sorted_correlations_df = pd.DataFrame(sorted_correlations, columns=['Variable', 'Correlation'])
    colors = ['green' if c > 0 else 'red' for c in sorted_correlations_df['Correlation']]
    
    axes[0].barh(sorted_correlations_df['Variable'], sorted_correlations_df['Correlation'], color=colors, alpha=0.7)
    axes[0].set_xlabel('Correlation with Price', fontsize=12)
    axes[0].set_title('Correlation of Variables with Price', fontsize=14, fontweight='bold')
    axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (var, corr) in enumerate(sorted_correlations):
        if not np.isnan(corr):
            axes[0].text(corr, i, f'  {corr:.4f}', va='center', fontsize=10)
    
    # Plot 2: Heatmap of full correlation matrix
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[1])
    axes[1].set_title('Full Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = DATA_DIR / 'correlation_analysis.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nCorrelation plots saved to: {plot_file}")
    
    # Display the plot
    plt.show()
    
    return correlation_matrix

def main():
    print("Loading properties data...")
    df = load_properties()
    
    print("Processing properties...")
    df_processed, removed_references = process_properties(df)
    
    # Save removed references to file
    if removed_references:
        removed_file = DATA_DIR / 'removed_references.txt'
        with open(removed_file, 'w') as f:
            f.write(f"# Properties removed during data cleaning\n")
            f.write(f"# Total removed: {len(removed_references)}\n")
            f.write(f"# Reason: Missing values in numeric columns (price, bedrooms, bathrooms, indoor_sqm)\n\n")
            for ref in removed_references:
                f.write(f"{ref}\n")
        print(f"\nRemoved references saved to: {removed_file}")
        print(f"Total references removed: {len(removed_references)}")
    
    print("Adding external data references...")
    df_processed = add_external_data(df_processed)
    
    print("Encoding categorical features...")
    df_processed = encode_categorical_features(df_processed)
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    stats = get_summary_stats(df_processed)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:,.2f}")
        else:
            print(f"{key}: {value:,}")
    
    print("\n" + "="*60)
    print("SAMPLE DATA (first 5 properties)")
    print("="*60)
    print(df_processed.head()[['reference', 'location', 'price', 'bedrooms', 'bathrooms']])
    
    # Analyze features
    feature_counts = analyze_features(df_processed)
    
    # Create binary feature columns - keep only top 20 by correlation with price
    df_with_features, valid_features = create_feature_columns(df_processed, min_count=100, top_n=20, by_correlation=True)
    
    # Variable analysis report
    generate_variable_report(df_with_features)
    
    # Variable distribution plots
    plot_variable_distributions(df_with_features)
    
    # Correlation analysis
    analyze_correlations(df_with_features)
    
    # Save processed data
    output_file = DATA_DIR / 'properties_processed.csv'
    # Drop the original features, features_list, and title columns before saving
    columns_to_drop = ['features', 'features_list', 'title']
    df_to_save = df_with_features.drop(columns=columns_to_drop)
    df_to_save.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")
    
    return df_with_features

if __name__ == "__main__":
    df = main()
