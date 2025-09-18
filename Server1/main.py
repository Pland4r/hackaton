from flask import Flask, jsonify,request
from flask_cors import CORS
from pymongo import MongoClient
import os
from routes.ml import ml_bp


from dotenv import load_dotenv
# import google.generativeai as genai  # REMOVE THIS
from helper.tool import (
check_job_fit, predict_salary,get_candidate,get_job,
screen_resume, get_priority, list_candidates, list_jobs
)
#from controllers.candidate_controller import register_candidate_routes
#import controllers.job_controller as job_controller




# Remove all Gemini/genai configuration and tool code


load_dotenv()

print("Environment variables:")
print(f"User: {os.environ.get('user_pass')}")
print(f"Password: {'*' * len(os.environ.get('pass_key', ''))}") 

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# MongoDB Atlas connection setup
def get_mongodb_connection():
    try:
        # Get credentials from environment variables
        username = os.environ.get('user_pass')
        password = os.environ.get('pass_key')
        
        if not username or not password:
            raise ValueError("MongoDB credentials not found in environment variables")
        
        # Alternative connection string format
        mongodb_uri = f"mongodb+srv://{username}:{password}@cluster0.ggtqgzr.mongodb.net/?retryWrites=true&w=majority"
        
        print(f"Attempting to connect with user: {username}")
        
        # Connect to MongoDB Atlas
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        
        # Test the connection by listing databases
        databases = client.list_database_names()
        print(f"Available databases: {databases}")
        
        # Check if flask_shop exists, if not create it
        if "flask_shop" in databases:
            db = client.flask_shop
            print("✅ Found flask_shop database!")
        else:
            print("❌ flask_shop database not found. Creating it...")
            db = client.flask_shop
            # Create a test collection to initialize the database
            db.test.insert_one({"test": "connection"})
            db.test.delete_one({"test": "connection"})
        
        return db
        
    except Exception as e:
        print(f"❌ Error connecting to MongoDB Atlas: {e}")
        return None

# Initialize MongoDB connection
db = get_mongodb_connection()
# mongo = PyMongo(app)


#chat route

# Remove all code related to genai, tools, and chat endpoint

# Health check for chat API
@app.route('/api/health', methods=['GET'])
def chat_health():
    """Health check endpoint for the chat API"""
    return jsonify({
        "status": "healthy",
        "service": "chat-api",
        "message": "Chat API is running"
    })

#call route


#register_candidate_routes(app, db)
#job_controller.register_job_routes(app, db)



#ml route
app.register_blueprint(ml_bp, url_prefix="/api")


# Home route
@app.route('/')
def home():
    if db is not None:  # FIXED: Compare with None instead of using if db:
        # Test connection again to make sure it's still alive
        try:
            db.command('ping')
            return jsonify({
                "message": "Welcome to the Flask backend with MongoDB Atlas!",
                "database": "connected",
                "status": "healthy",
                "connection_type": "MongoDB Atlas",
                "cluster": "Cluster0"
            })
        except Exception as e:
            return jsonify({
                "message": "Welcome to the Flask backend!",
                "database": "disconnected",
                "status": "unhealthy",
                "error": str(e)
            })
    else:
        return jsonify({
            "message": "Welcome to the Flask backend!",
            "database": "disconnected",
            "status": "unhealthy",
            "connection_type": "MongoDB Atlas"
        })

if __name__ == '__main__':
    # Get port from environment variable (for cloud deployment)
    port = int(os.environ.get('PORT', 5000))
    
    # Print connection status on startup
    if db is not None:
        print("🚀 Server starting with MongoDB Atlas connection")
        print("🌐 Cluster: Cluster0")
        print("🔗 Connection URI: mongodb+srv://cluster0.ggtqgzr.mongodb.net/")
    else:
        print("⚠️  Server starting without MongoDB connection")
    
    # Run the app
    print(f"🌐 Server running on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)