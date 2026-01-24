import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv('NEON_DATABASE_URL')

def fix_table():
    """Fix the OLTP table to handle data properly"""
    
    print("🔧 Fixing OLTP table schema...")
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # Drop existing table
    print("   Dropping old table...")
    cur.execute("DROP TABLE IF EXISTS url_data_oltp;")
    
    # Create new table with better schema
    print("   Creating new table...")
    cur.execute("""
        CREATE TABLE url_data_oltp (
            id SERIAL PRIMARY KEY,
            url TEXT NOT NULL,
            type VARCHAR(200) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Create indexes
    cur.execute("""
        CREATE INDEX idx_url_type ON url_data_oltp(type);
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    
    print("✅ Table created successfully!")

if __name__ == "__main__":
    fix_table()
