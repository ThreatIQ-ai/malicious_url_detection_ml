import pandas as pd
import psycopg2
import pyarrow as pa
import pyarrow.parquet as pq
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
DATABASE_URL = os.getenv('NEON_DATABASE_URL')

def export_to_parquet():
    """Export Neon data directly to Parquet - No transformations"""
    
    print("=" * 60)
    print("-- Exporting OLTP (Neon) → Parquet")
    print("=" * 60)
    
    # Connect to Neon
    print("\n-- Connecting to Neon...")
    conn = psycopg2.connect(DATABASE_URL)
    
    # Extract all data
    print("-- Extracting data from Neon...")
    query = "SELECT * FROM url_data_oltp ORDER BY id;"
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"-- Extracted {len(df):,} records")
    print(f"   Columns: {list(df.columns)}")
    
    # Show distribution
    print(f"\n-- Type Distribution:")
    for url_type, count in df['type'].value_counts().items():
        print(f"   {url_type:15} : {count:,}")
    
    # Create output directory
    output_dir = 'parquet_files'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save as single Parquet file
    print(f"\n-- Saving to Parquet...")
    output_file = f"{output_dir}/malicious_urls_{timestamp}.parquet"
    
    table = pa.Table.from_pandas(df)
    pq.write_table(
        table, 
        output_file, 
        compression='snappy'
    )
    
    # Get file size
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    
    print(f"-- Parquet file created!")
    print(f"   File: {output_file}")
    print(f"   Size: {file_size:.2f} MB")
    print(f"   Records: {len(df):,}")
    
    print(f"\n-- File location: {os.path.abspath(output_file)}")
    
    return output_file

if __name__ == "__main__":
    export_to_parquet()