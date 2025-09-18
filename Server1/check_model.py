import pickle

def check_model_file(path):
    """Check what's in the model file"""
    try:
        with open(path, "rb") as f:
            content = pickle.load(f)
        
        print(f"Type of content: {type(content)}")
        
        if isinstance(content, dict):
            print("Dictionary contents:")
            for key, value in content.items():
                print(f"  {key}: {type(value)}")
                if hasattr(value, 'predict'):
                    print(f"    -> Has predict method!")
        
        return content
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    model_path = "D:/hackathon-main/hackathon-main/server/models/linear_regression.pkl"
    print("Checking linear regression model...")
    content = check_model_file(model_path)
    
    print("\n" + "="*50 + "\n")
    
    model_path = "D:/hackathon-main/hackathon-main/server/models/random_forest.pkl"
    print("Checking random forest model...")
    content = check_model_file(model_path)