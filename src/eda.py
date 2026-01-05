# src/eda.py
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
import logging
from tqdm import tqdm
import warnings
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import ssl

# Remove the entire try-catch block that checks for NLTK data
# Disable SSL verification for NLTK download (common issue on some systems)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data if not already present - SIMPLIFIED
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
        print("NLTK data already available")
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("NLTK data downloaded")

# Call it once when module loads
download_nltk_data()

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_complaints(file_path: str, chunksize: Optional[int] = 100000, 
                    usecols: Optional[List[str]] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load CFPB complaints dataset efficiently
    
    Args:
        file_path: Path to CSV file
        chunksize: Number of rows per chunk (for large files)
        usecols: Specific columns to load (saves memory)
        nrows: Number of rows to load (for testing)
    
    Returns:
        pd.DataFrame: Loaded complaints data
    """
    try:
        file_size = os.path.getsize(file_path) / (1024**3)  # Size in GB
        logger.info(f"Loading dataset from: {file_path}")
        logger.info(f"File size: {file_size:.2f} GB")
        
        if nrows:
            logger.info(f"Loading first {nrows} rows for testing...")
            df = pd.read_csv(file_path, nrows=nrows, usecols=usecols, low_memory=False)
        elif file_size > 1.0:  # If file is larger than 1GB, use chunking
            logger.info(f"File is large ({file_size:.2f} GB). Loading in chunks...")
            
            # Read in chunks
            chunks = []
            chunk_reader = pd.read_csv(file_path, chunksize=chunksize, usecols=usecols, low_memory=False)
            
            for chunk in tqdm(chunk_reader, desc="Loading chunks"):
                chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True)
        else:
            # Load normally for smaller files
            df = pd.read_csv(file_path, usecols=usecols, low_memory=False)
        
        logger.info(f"✅ Successfully loaded {len(df):,} rows and {len(df.columns)} columns")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error loading {file_path}: {str(e)}")
        raise

def initial_eda(df: pd.DataFrame, narrative_col: str = 'Consumer complaint narrative') -> Dict:
    """
    Perform initial exploratory data analysis
    
    Args:
        df: Input dataframe
        narrative_col: Name of narrative column
    
    Returns:
        Dict: EDA statistics
    """
    logger.info("📊 Performing Initial EDA...")
    logger.info("-" * 40)
    
    # Basic info
    logger.info(f"Dataset shape: {df.shape} (rows, columns)")
    logger.info(f"\nColumns: {list(df.columns)}")
    logger.info(f"\nData types:\n{df.dtypes}")
    
    # Product distribution
    logger.info(f"\n📈 Top 10 Products in Full Dataset:")
    product_dist = df['Product'].value_counts().head(10)
    for product, count in product_dist.items():
        percentage = count / len(df) * 100
        logger.info(f"  {product}: {count:,} complaints ({percentage:.1f}%)")
    
    # Narrative analysis
    has_narrative = df[narrative_col].notna() & (df[narrative_col].str.strip() != '')
    narratives_count = has_narrative.sum()
    no_narratives_count = len(df) - narratives_count
    
    logger.info(f"\n📝 Narrative Analysis:")
    logger.info(f"  Complaints WITH narrative: {narratives_count:,} ({narratives_count/len(df)*100:.1f}%)")
    logger.info(f"  Complaints WITHOUT narrative: {no_narratives_count:,} ({no_narratives_count/len(df)*100:.1f}%)")
    
    # Word count analysis for narratives
    if narratives_count > 0:
        df_narratives = df[has_narrative].copy()
        df_narratives['word_count'] = df_narratives[narrative_col].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        word_stats = df_narratives['word_count'].describe()
        
        logger.info(f"\n📊 Word Count Statistics (narratives only):")
        logger.info(f"  Mean: {word_stats['mean']:.1f} words")
        logger.info(f"  Std: {word_stats['std']:.1f} words")
        logger.info(f"  Min: {word_stats['min']:.0f} words")
        logger.info(f"  25th percentile: {word_stats['25%']:.0f} words")
        logger.info(f"  50th percentile (median): {word_stats['50%']:.0f} words")
        logger.info(f"  75th percentile: {word_stats['75%']:.0f} words")
        logger.info(f"  Max: {word_stats['max']:.0f} words")
        
        # Identify very short and long narratives
        short_threshold = 10
        long_threshold = 500
        
        short_narratives = (df_narratives['word_count'] < short_threshold).sum()
        long_narratives = (df_narratives['word_count'] > long_threshold).sum()
        
        logger.info(f"\n⚠️  Narratives with < {short_threshold} words: {short_narratives:,} ({short_narratives/len(df_narratives)*100:.1f}%)")
        logger.info(f"⚠️  Narratives with > {long_threshold} words: {long_narratives:,} ({long_narratives/len(df_narratives)*100:.1f}%)")
    
    return {
        'total_rows': len(df),
        'narratives_count': narratives_count,
        'no_narratives_count': no_narratives_count,
        'product_distribution': product_dist.to_dict()
    }

def filter_by_products(df: pd.DataFrame, target_products: List[str]) -> pd.DataFrame:
    """
    Filter dataframe to include only specified products
    
    Args:
        df: Input dataframe
        target_products: List of products to include
    
    Returns:
        Filtered dataframe
    """
    logger.info("\n🔍 Filtering by Target Products...")
    logger.info("-" * 40)
    
    original_shape = df.shape
    logger.info(f"Original dataset shape: {original_shape}")
    
    # Clean product names for consistent filtering
    df['Product_clean'] = df['Product'].str.lower().str.strip().fillna('')
    target_products_clean = [p.lower().strip() for p in target_products]
    
    # Filter for target products
    mask = df['Product_clean'].isin(target_products_clean)
    filtered_df = df[mask].copy()
    
    logger.info(f"After product filtering: {filtered_df.shape}")
    logger.info(f"Removed {original_shape[0] - len(filtered_df):,} rows")
    
    # Show distribution
    if len(filtered_df) > 0:
        product_counts = filtered_df['Product'].value_counts()
        logger.info("\n📊 Product Distribution After Filtering:")
        for product, count in product_counts.items():
            percentage = count / len(filtered_df) * 100
            logger.info(f"  {product}: {count:,} complaints ({percentage:.1f}%)")
    else:
        logger.warning("⚠️  No complaints found for target products!")
    
    return filtered_df

def remove_empty_narratives(df: pd.DataFrame, narrative_col: str = 'Consumer complaint narrative') -> pd.DataFrame:
    """
    Remove records with empty narrative fields
    
    Args:
        df: Input dataframe
        narrative_col: Name of narrative column
    
    Returns:
        Dataframe with non-empty narratives
    """
    logger.info("\n🧹 Removing Empty Narratives...")
    logger.info("-" * 40)
    
    original_count = len(df)
    
    # Identify empty narratives
    is_empty = df[narrative_col].isna() | (df[narrative_col].str.strip() == '')
    empty_count = is_empty.sum()
    
    # Filter out empty narratives
    filtered_df = df[~is_empty].copy()
    
    logger.info(f"Original complaints: {original_count:,}")
    logger.info(f"Empty narratives removed: {empty_count:,}")
    logger.info(f"Remaining complaints: {len(filtered_df):,}")
    logger.info(f"Percentage retained: {len(filtered_df)/original_count*100:.1f}%")
    
    return filtered_df

def clean_text_noise(text: str) -> str:
    """
    Clean and normalize complaint narrative text
    
    Args:
        text: Raw complaint narrative
    
    Returns:
        Cleaned text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Step 1: Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Step 2: Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Step 3: Remove phone numbers (US format)
    text = re.sub(r'\b\d{3}[-.\s]?\d{4}\b', '', text)  # 7 digit format
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '', text)  # 10 digit format
    
    # Step 4: Remove IBAN/transaction IDs
    text = re.sub(r'\b[A-Z]{2}\d{10,}\b', '', text, flags=re.IGNORECASE)
    
    # Step 5: Remove common boilerplate phrases
    boilerplate_phrases = [
        r'i am writing to file a complaint',
        r'i would like to file a complaint',
        r'this is a complaint regarding',
        r'dear sir/madam',
        r'to whom it may concern',
        r'please be advised that',
        r'i am writing to express my dissatisfaction',
        r'i am writing to report an issue'
    ]
    
    for phrase in boilerplate_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)
    
    # Step 6: Remove special characters but keep basic punctuation and apostrophes
    text = re.sub(r'[^\w\s\'-]', '', text)
    
    # Step 7: Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Step 8: Remove standalone single quotes
    text = re.sub(r'\s\'|\'\s', ' ', text)
    
    return text.strip()

def normalize_text(text: str) -> str:
    """
    Apply NLP normalization: tokenization, stopword removal, lemmatization
    
    Args:
        text: Cleaned text
    
    Returns:
        Normalized text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Initialize NLTK components
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # Apply lemmatization for verbs
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    
    # Apply lemmatization for nouns (default)
    tokens = [lemmatizer.lemmatize(token, pos='n') for token in tokens]
    
    # Reconstruct text
    return " ".join(tokens)

def create_visualizations(df: pd.DataFrame, output_dir: str = 'notebooks/figures'):
    """
    Create EDA visualizations
    
    Args:
        df: Dataframe with complaint data
        output_dir: Directory to save figures
    """
    logger.info("\n🎨 Creating Visualizations...")
    logger.info("-" * 40)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Product Distribution
    plt.figure(figsize=(14, 7))
    product_counts = df['Product'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(product_counts)))
    
    bars = plt.bar(product_counts.index, product_counts.values, color=colors, edgecolor='black')
    plt.title('Distribution of Complaints by Product', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Product', fontsize=12)
    plt.ylabel('Number of Complaints', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'product_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Word Count Distribution (if word_count column exists)
    if 'word_count' in df.columns and df['word_count'].sum() > 0:
        plt.figure(figsize=(15, 5))
        
        # Histogram
        plt.subplot(1, 2, 1)
        narratives_df = df[df['word_count'] > 0]
        plt.hist(narratives_df['word_count'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        plt.title('Distribution of Word Counts', fontsize=14, fontweight='bold')
        plt.xlabel('Word Count', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.axvline(narratives_df['word_count'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {narratives_df["word_count"].mean():.1f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        box = plt.boxplot(narratives_df['word_count'], vert=False, patch_artist=True)
        box['boxes'][0].set_facecolor('lightgreen')
        plt.title('Word Count Box Plot', fontsize=14, fontweight='bold')
        plt.xlabel('Word Count', fontsize=12)
        plt.yticks([])
        
        # Add statistics text
        stats_text = f"""
        Min: {narratives_df['word_count'].min():.0f}
        25th: {narratives_df['word_count'].quantile(0.25):.0f}
        Median: {narratives_df['word_count'].median():.0f}
        75th: {narratives_df['word_count'].quantile(0.75):.0f}
        Max: {narratives_df['word_count'].max():.0f}
        """
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'word_count_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. Missing Values Heatmap
    plt.figure(figsize=(12, 6))
    missing_data = df.isnull().mean().sort_values(ascending=False)
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) > 0:
        colors = ['#ff6b6b' if x > 0.5 else '#feca57' if x > 0.1 else '#1dd1a1' for x in missing_data]
        bars = plt.bar(missing_data.index, missing_data.values * 100, color=colors, edgecolor='black')
        plt.title('Percentage of Missing Values by Column', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Column', fontsize=12)
        plt.ylabel('Percentage Missing (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 105)
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'missing_values.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    logger.info(f"✅ Visualizations saved to: {output_dir}")

def perform_complete_preprocessing(df: pd.DataFrame, target_products: List[str]) -> pd.DataFrame:
    """
    Complete preprocessing pipeline
    
    Args:
        df: Raw complaints dataframe
        target_products: List of target products
    
    Returns:
        Cleaned and preprocessed dataframe
    """
    logger.info("\n🔄 Starting Complete Preprocessing Pipeline...")
    logger.info("=" * 60)
    
    # Step 1: Filter by products
    filtered_df = filter_by_products(df, target_products)
    
    if len(filtered_df) == 0:
        logger.error("❌ No data after product filtering. Exiting pipeline.")
        return pd.DataFrame()
    
    # Step 2: Remove empty narratives
    filtered_df = remove_empty_narratives(filtered_df)
    
    if len(filtered_df) == 0:
        logger.error("❌ No data after removing empty narratives. Exiting pipeline.")
        return pd.DataFrame()
    
    # Step 3: Clean text (remove HTML, URLs, phone numbers, etc.)
    logger.info("\n🧼 Cleaning Text (HTML, URLs, PII)...")
    filtered_df['cleaned_narrative'] = filtered_df['Consumer complaint narrative'].apply(clean_text_noise)
    
    # Step 4: Normalize text (tokenization, stopword removal, lemmatization)
    logger.info("\n🔧 Normalizing Text (Tokenization, Stopwords, Lemmatization)...")
    filtered_df['normalized_narrative'] = filtered_df['cleaned_narrative'].apply(normalize_text)
    
    # Step 5: Add word count for normalized narratives
    filtered_df['word_count'] = filtered_df['normalized_narrative'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) and str(x).strip() != '' else 0
    )
    
    # Step 6: Remove extremely short narratives
    min_words = 3
    before_filter = len(filtered_df)
    filtered_df = filtered_df[filtered_df['word_count'] >= min_words].copy()
    removed_short = before_filter - len(filtered_df)
    
    if removed_short > 0:
        logger.info(f"Removed {removed_short:,} narratives with < {min_words} words")
    
    # Final statistics
    logger.info("\n✅ Preprocessing Complete!")
    logger.info("=" * 60)
    logger.info(f"Final dataset shape: {filtered_df.shape}")
    logger.info(f"Total complaints processed: {len(df):,}")
    logger.info(f"Complaints after preprocessing: {len(filtered_df):,}")
    logger.info(f"Percentage retained: {len(filtered_df)/len(df)*100:.2f}%")
    
    # Word count statistics
    if 'word_count' in filtered_df.columns:
        word_stats = filtered_df['word_count'].describe()
        logger.info(f"\n📊 Final Word Count Statistics:")
        logger.info(f"  Mean: {word_stats['mean']:.1f} words")
        logger.info(f"  Min: {word_stats['min']:.0f} words")
        logger.info(f"  Max: {word_stats['max']:.0f} words")
        logger.info(f"  Std: {word_stats['std']:.1f} words")
    
    return filtered_df

def save_filtered_complaints(df: pd.DataFrame, output_path: str):
    """
    Save filtered and cleaned complaints to CSV
    
    Args:
        df: Cleaned dataframe
        output_path: Output file path
    """
    logger.info(f"\n💾 Saving cleaned data to: {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Select columns to save
    columns_to_save = [
        'Complaint ID',
        'Product',
        'Sub-product',
        'Issue',
        'Sub-issue',
        'Consumer complaint narrative',  # Original
        'cleaned_narrative',             # After basic cleaning
        'normalized_narrative',          # After NLP normalization
        'word_count',                    # Word count of normalized narrative
        'Company',
        'State',
        'ZIP code',
        'Date received',
        'Date sent to company',
        'Company response to consumer',
        'Consumer disputed?',
        'Tags',
        'Consumer consent provided?',
        'Submitted via'
    ]
    
    # Select only existing columns
    existing_columns = [col for col in columns_to_save if col in df.columns]
    df_to_save = df[existing_columns].copy()
    
    # Save to CSV
    df_to_save.to_csv(output_path, index=False)
    
    file_size = os.path.getsize(output_path) / (1024**2)  # Size in MB
    logger.info(f"✅ Saved {len(df_to_save):,} complaints to {output_path}")
    logger.info(f"File size: {file_size:.2f} MB")

def run_complete_pipeline(input_path: str, output_path: str, nrows: Optional[int] = None):
    """
    Run complete EDA and preprocessing pipeline
    
    Args:
        input_path: Path to raw complaints CSV
        output_path: Path to save cleaned data
        nrows: Number of rows to load (for testing)
    """
    logger.info("=" * 80)
    logger.info("🚀 STARTING COMPLETE EDA AND PREPROCESSING PIPELINE")
    logger.info("=" * 80)
    
    # Target products from Task 1
    TARGET_PRODUCTS = [
        'Credit card',
        'Personal loan',
        'Buy Now, Pay Later (BNPL)',
        'Savings account',
        'Money transfers'
    ]
    
    # Essential columns needed for analysis
    essential_columns = [
        'Complaint ID',
        'Product',
        'Sub-product',
        'Issue',
        'Sub-issue',
        'Consumer complaint narrative',
        'Company',
        'State',
        'ZIP code',
        'Date received',
        'Date sent to company',
        'Company response to consumer',
        'Consumer disputed?',
        'Tags',
        'Consumer consent provided?',
        'Submitted via'
    ]
    
    try:
        # Step 1: Load data
        logger.info("\n📥 STEP 1: LOADING DATA")
        df = load_complaints(input_path, usecols=essential_columns, nrows=nrows)
        
        # Step 2: Initial EDA
        logger.info("\n📊 STEP 2: INITIAL EXPLORATORY DATA ANALYSIS")
        eda_results = initial_eda(df)
        
        # Step 3: Preprocessing
        logger.info("\n🔧 STEP 3: PREPROCESSING")
        cleaned_df = perform_complete_preprocessing(df, TARGET_PRODUCTS)
        
        if len(cleaned_df) == 0:
            logger.error("❌ Pipeline failed: No data after preprocessing")
            return
        
        # Step 4: Create visualizations
        logger.info("\n🎨 STEP 4: CREATING VISUALIZATIONS")
        create_visualizations(cleaned_df)
        
        # Step 5: Save cleaned data
        logger.info("\n💾 STEP 5: SAVING CLEANED DATA")
        save_filtered_complaints(cleaned_df, output_path)
        
        # Step 6: Final report
        logger.info("\n📋 STEP 6: FINAL SUMMARY")
        logger.info("=" * 60)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        logger.info(f"\n📈 Key Statistics:")
        logger.info(f"  • Total complaints loaded: {len(df):,}")
        logger.info(f"  • Complaints with narratives: {eda_results['narratives_count']:,}")
        logger.info(f"  • Complaints after filtering: {len(cleaned_df):,}")
        logger.info(f"  • Retention rate: {len(cleaned_df)/len(df)*100:.2f}%")
        
        logger.info(f"\n🎯 Target Products Distribution:")
        product_counts = cleaned_df['Product'].value_counts()
        for product, count in product_counts.items():
            percentage = count / len(cleaned_df) * 100
            logger.info(f"  • {product}: {count:,} ({percentage:.1f}%)")
        
        logger.info(f"\n📝 Narrative Statistics:")
        if 'word_count' in cleaned_df.columns:
            avg_words = cleaned_df['word_count'].mean()
            median_words = cleaned_df['word_count'].median()
            logger.info(f"  • Average words per narrative: {avg_words:.1f}")
            logger.info(f"  • Median words per narrative: {median_words:.1f}")
        
        logger.info(f"\n💾 Output saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()