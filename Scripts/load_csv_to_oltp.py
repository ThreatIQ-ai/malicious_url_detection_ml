import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import os
from dotenv import load_dotenv
from tqdm import tqdm
import chardet

load_dotenv()
DATABASE_URL = os.getenv('NEON_DATABASE_URL')

def detect_encoding(csv_filename):
    """Detect the encoding of the CSV file"""
    print(f"-- Detecting file encoding...")
    
    with open(csv_filename, 'rb') as f:
        result = chardet.detect(f.read(100000))
    
    encoding = result['encoding']
    confidence = result['confidence']
    
    print(f"   Detected: {encoding} (confidence: {confidence:.2%})")
    return encoding

def clean_data(df):
    """Clean and validate the data"""
    print(f" Cleaning data...")
    
    original_count = len(df)
    
    # Keep only 'url' and 'type' columns
    df = df[['url', 'type']].copy()
    
    # Remove rows with missing values
    df = df.dropna()
    print(f"   Removed {original_count - len(df)} rows with missing values")
    
    # Clean URLs (convert to string and strip whitespace)
    df['url'] = df['url'].astype(str).str.strip()
    
    df['type'] = df['type'].astype(str).str.strip().str.lower()
    
    valid_types = ['benign', 'phishing', 'defacement', 'malware']
    
    # Mark suspicious types
    df['is_valid_type'] = df['type'].isin(valid_types)
    
    # Count invalid types
    invalid_count = (~df['is_valid_type']).sum()
    if invalid_count > 0:
        print(f"   Found {invalid_count} rows with invalid types")
        print(f"   Invalid types will be marked as 'unknown'")
        
        # Show sample of invalid types
        invalid_types = df[~df['is_valid_type']]['type'].value_counts().head(10)
        print(f"\n   Top invalid types:")
        for type_val, count in invalid_types.items():
            print(f"      - '{type_val}': {count} occurrences")
    
    # Replace invalid types with 'unknown'
    df.loc[~df['is_valid_type'], 'type'] = 'unknown'
    
    # Drop the helper column
    df = df.drop('is_valid_type', axis=1)
    
    # Remove duplicate URLs (keep first occurrence)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['url'], keep='first')
    duplicates_removed = before_dedup - len(df)
    if duplicates_removed > 0:
        print(f"   Removed {duplicates_removed} duplicate URLs")
    
    # Remove URLs that are too short or too long
    df = df[(df['url'].str.len() > 5) & (df['url'].str.len() < 2048)]
    
    print(f"- Cleaning complete!")
    print(f"   Final record count: {len(df):,}")
    
    return df

def load_csv_to_oltp(csv_filename, batch_size=5000):
    """Load CSV data into OLTP database"""
    
    print(f"-- Reading CSV file: {csv_filename}")
    
    try:
        # Detect encoding
        encoding = detect_encoding(csv_filename)
        
        # Read CSV file with detected encoding
        print(f"-- Reading CSV with {encoding} encoding...")
        
        # Read only first 2 columns to avoid unnamed columns
        df = pd.read_csv(
            csv_filename, 
            encoding=encoding, 
            encoding_errors='replace',
            usecols=[0, 1],  # Only read first 2 columns
            low_memory=False
        )
        
        print(f"-- CSV loaded successfully!")
        print(f"   Total records: {len(df):,}")
        print(f"   Columns: {list(df.columns)}")
        
        # Rename columns to ensure consistency
        df.columns = ['url', 'type']
        
        # Clean the data
        df = clean_data(df)
        
        # Show distribution
        print(f"\n-- URL Type Distribution (after cleaning):")
        type_counts = df['type'].value_counts()
        for type_val, count in type_counts.items():
            print(f"   {type_val:15} : {count:,}")
        
        # Connect to database
        print(f"\n-- Connecting to Neon database...")
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Clear existing data (optional - comment out if you want to keep old data)
        print(f"--  Clearing existing data...")
        cur.execute("TRUNCATE TABLE url_data_oltp RESTART IDENTITY;")
        conn.commit()
        
        # Load data in batches
        print(f"\n-- Loading data in batches of {batch_size:,}...")
        total_records = len(df)
        total_inserted = 0
        errors = 0
        
        for i in tqdm(range(0, total_records, batch_size), desc="Progress"):
            # Get batch
            batch = df.iloc[i:i+batch_size]
            
            # Prepare records
            records = []
            for _, row in batch.iterrows():
                try:
                    url = str(row['url'])[:2000]  # Limit URL length
                    url_type = str(row['type'])[:200]  # Limit type length
                    records.append((url, url_type))
                except Exception as e:
                    errors += 1
                    continue
            
            # Insert batch
            if records:
                insert_query = """
                    INSERT INTO url_data_oltp (url, type)
                    VALUES (%s, %s);
                """
                
                try:
                    execute_batch(cur, insert_query, records, page_size=1000)
                    conn.commit()
                    total_inserted += len(records)
                except Exception as e:
                    conn.rollback()
                    print(f"\n  Error in batch: {e}")
                    errors += len(records)
        
        print(f"\n-- Data loaded successfully!")
        print(f"   Total records inserted: {total_inserted:,}")
        if errors > 0:
            print(f"   Errors/Skipped: {errors}")
        
        # Verify
        cur.execute("SELECT COUNT(*) FROM url_data_oltp;")
        count = cur.fetchone()[0]
        print(f"   Records in database: {count:,}")
        
        # Show final distribution in database
        print(f"\n Final Distribution in Database:")
        cur.execute("""
            SELECT type, COUNT(*) as count 
            FROM url_data_oltp 
            GROUP BY type 
            ORDER BY count DESC;
        """)
        
        for row in cur.fetchall():
            print(f"   {row[0]:15} : {row[1]:,}")
        
        cur.close()
        conn.close()
        
    except FileNotFoundError:
        print(f" Error: File '{csv_filename}' not found!")
        print(f"   Make sure the CSV file is in the same folder.")
        print(f"   Current folder files:")
        for file in os.listdir('.'):
            if file.endswith('.csv'):
                print(f"   - {file}")
                
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("Loading CSV Data to OLTP (Neon)")
    print("=" * 60)
    
    csv_file = "malicious_urls.csv"
    
    load_csv_to_oltp(csv_file)
