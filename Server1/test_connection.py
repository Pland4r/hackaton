from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def test_connection():
    try:
        username = "bmciconnect3_db_user"
        password = "OeGsFz8WMfWerWsx"
        
        print(f"Testing connection with: {username}:{password}")
        
        # Try different connection formats
        mongodb_uri = f"mongodb+srv://{username}:{password}@cluster0.ggtqgzr.mongodb.net/admin?retryWrites=true&w=majority"
        
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=10000)
        
        # Test the connection
        client.admin.command('ping')
        print("✅ Connection successful!")
        
        # List all databases
        databases = client.list_database_names()
        print(f"Available databases: {databases}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()