import pandas as pd
import os
from pathlib import Path

# File paths
DATA_DIR = Path('data')
CSV_FILE = DATA_DIR / 'properties.csv'
DESCRIPTIONS_DIR = DATA_DIR / 'descriptions' / 'descriptions'
IMAGES_DIR = DATA_DIR / 'images' / 'images'

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
    """Load the properties CSV file into a pandas DataFrame."""
    df = pd.read_csv(CSV_FILE, names=COLUMNS, encoding='utf-8')
    print(f"Loaded {len(df)} properties")
    return df

def clean_price(price_str):
    """Convert price string (e.g., '€450,000') to numeric value."""
    if pd.isna(price_str):
        return None
    # Remove currency symbol and commas
    price_clean = str(price_str).replace('€', '').replace(',', '').strip()
    try:
        return float(price_clean)
    except ValueError:
        return None

def parse_features(features_str):
    """Parse pipe-separated features into a list."""
    if pd.isna(features_str) or features_str == '':
        return []
    return [f.strip() for f in str(features_str).split('|')]

def process_properties(df):
    """Clean and process the properties DataFrame."""
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Clean price column
    df_processed['price_numeric'] = df_processed['price'].apply(clean_price)
    
    # Parse features into lists
    df_processed['features_list'] = df_processed['features'].apply(parse_features)
    df_processed['num_features'] = df_processed['features_list'].apply(len)
    
    # Convert numeric columns (handle 'NA' strings)
    numeric_cols = ['bedrooms', 'bathrooms', 'indoor_sqm', 'outdoor_sqm']
    for col in numeric_cols:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Extract property type from title
    df_processed['property_type'] = df_processed['title'].str.extract(r'(\d+\s+Bedroom\s+\w+|\w+\s+Villa|\w+|Plot)', expand=False)
    
    return df_processed

def get_description(reference):
    """Load description text for a given property reference."""
    desc_file = DESCRIPTIONS_DIR / f"{reference}.txt"
    if desc_file.exists():
        with open(desc_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None

def get_images(reference):
    """Get list of image files for a given property reference."""
    images_folder = IMAGES_DIR / reference
    if images_folder.exists() and images_folder.is_dir():
        return list(images_folder.glob('*'))
    return []

def add_external_data(df):
    """Add description and image count to the DataFrame."""
    df['has_description'] = df['reference'].apply(
        lambda ref: (DESCRIPTIONS_DIR / f"{ref}.txt").exists()
    )
    df['image_count'] = df['reference'].apply(
        lambda ref: len(list((IMAGES_DIR / ref).glob('*'))) 
        if (IMAGES_DIR / ref).exists() else 0
    )
    return df

def get_summary_stats(df):
    """Generate summary statistics for the dataset."""
    stats = {
        'total_properties': len(df),
        'avg_price': df['price_numeric'].mean(),
        'median_price': df['price_numeric'].median(),
        'min_price': df['price_numeric'].min(),
        'max_price': df['price_numeric'].max(),
        'avg_bedrooms': df['bedrooms'].mean(),
        'avg_bathrooms': df['bathrooms'].mean(),
        'avg_indoor_sqm': df['indoor_sqm'].mean(),
        'properties_with_outdoor': df['outdoor_sqm'].notna().sum(),
        'avg_features': df['num_features'].mean(),
        'properties_with_description': df['has_description'].sum(),
        'total_images': df['image_count'].sum()
    }
    return stats

def main():
    """Main processing pipeline."""
    print("Loading properties data...")
    df = load_properties()
    
    print("Processing properties...")
    df_processed = process_properties(df)
    
    print("Adding external data references...")
    df_processed = add_external_data(df_processed)
    
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
    print(df_processed.head()[['reference', 'location', 'price_numeric', 'bedrooms', 'bathrooms', 'num_features']])
    
    # Save processed data
    output_file = DATA_DIR / 'properties_processed.csv'
    df_processed.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")
    
    return df_processed

if __name__ == "__main__":
    df = main()
