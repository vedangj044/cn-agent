import sqlite3
import qdrant_client
from qdrant_client.http import models
import json
from datetime import datetime
import logging
from typing import Dict, Any, List
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def print_sqlite_contents():
    """Print contents of SQLite database tables"""
    logger.info("Connecting to SQLite database...")
    conn = sqlite3.connect('k8s_analysis.db')
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print("\n=== SQLite Database Contents ===")
    for table in tables:
        table_name = table[0]
        print(f"\n--- Table: {table_name} ---")
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Get table contents
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        
        if not rows:
            print("  No data")
            continue
            
        # Print column headers
        print("  | " + " | ".join(columns) + " |")
        print("  |" + "|".join(["-" * len(col) for col in columns]) + "|")
        
        # Print rows
        for row in rows:
            # Format each cell, handling None values and JSON strings
            formatted_cells = []
            for cell in row:
                if cell is None:
                    formatted_cells.append("NULL")
                elif isinstance(cell, str) and (cell.startswith('{') or cell.startswith('[')):
                    try:
                        # Pretty print JSON
                        formatted_cells.append(json.dumps(json.loads(cell), indent=2))
                    except:
                        formatted_cells.append(str(cell))
                else:
                    formatted_cells.append(str(cell))
            print("  | " + " | ".join(formatted_cells) + " |")
    
    conn.close()

def print_qdrant_contents():
    """Print contents of Qdrant collections"""
    logger.info("Connecting to Qdrant...")
    qdrant = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_URL", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333))
    )
    
    print("\n=== Qdrant Collections Contents ===")
    
    # Get all collections
    collections = qdrant.get_collections().collections
    
    for collection in collections:
        collection_name = collection.name
        print(f"\n--- Collection: {collection_name} ---")
        
        # Get collection info
        collection_info = qdrant.get_collection(collection_name=collection_name)
        print(f"  Vector Size: {collection_info.config.params.vectors.size}")
        print(f"  Distance: {collection_info.config.params.vectors.distance}")
        
        # Get all points
        try:
            points = qdrant.scroll(
                collection_name=collection_name,
                limit=100,  # Adjust as needed
                with_payload=True,
                with_vectors=False  # Don't print vectors to keep output readable
            )[0]
            
            if not points:
                print("  No data")
                continue
                
            # Print points
            for point in points:
                print(f"\n  Point ID: {point.id}")
                print("  Payload:")
                print(json.dumps(point.payload, indent=4))
                
        except Exception as e:
            logger.error(f"Error fetching points from collection {collection_name}: {str(e)}")

def main():
    """Main function to print database contents"""
    try:
        print_sqlite_contents()
        print_qdrant_contents()
    except Exception as e:
        logger.error(f"Error during database inspection: {str(e)}")
        raise

if __name__ == "__main__":
    main() 