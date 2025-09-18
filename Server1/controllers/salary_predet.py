import pandas as pd
import pickle


def load_model(path: str):
    try:
        with open(path, "rb") as f:
            model_data = pickle.load(f)
            
        print(f"Loaded object type: {type(model_data)}")
        
        # If it's a dictionary with a model inside, extract it
        if isinstance(model_data, dict) and 'model' in model_data:
            print("Found model in dictionary")
            return model_data
        else:
            print("Model format not recognized")
            return None
            
    except FileNotFoundError:
        print(f"❌ Model file not found at: {path}")
        return None
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        return None

def prepare_input(data: dict, model_data):
    """Prepare input dict using the preprocessing from the model dictionary"""
    # Extract components from the model data
    model = model_data['model']
    scaler = model_data.get('scaler')
    ordinal_maps = model_data.get('ordinal_maps', {})
    rare_categories = model_data.get('rare_categories', {})
    model_columns = model_data.get('model_columns', [])
    
    # Create DataFrame from input
    input_df = pd.DataFrame([data])
    
    # Apply preprocessing based on what's available in model_data
    # 1. Handle categorical encoding using ordinal_maps
    for col, mapping in ordinal_maps.items():
        if col in input_df.columns:
            # Map values, use a default for unseen categories
            input_df[col] = input_df[col].map(mapping).fillna(-1)
    
    # 2. Handle rare categories (replace with 'other' or similar)
    for col, rare_values in rare_categories.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].apply(lambda x: 'other' if x in rare_values else x)
    
    # 3. Ensure all expected columns are present
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # 4. Reorder columns to match training order
    if model_columns:
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # 5. Apply scaling if scaler is available
    if scaler is not None:
        try:
            input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
        except Exception as e:
            print(f"Warning: Scaling failed: {e}")
    
    return input_df, model

def predict(model_path: str, data: dict):
    """Load model, prepare input and return prediction."""
    model_data = load_model(model_path)
    if model_data is None:
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Check if we have a model in the dictionary
    if 'model' not in model_data:
        raise ValueError("No model found in the loaded data")
    
    model = model_data['model']
    
    # Prepare input data
    input_df, model = prepare_input(data, model_data)
    
    try:
        prediction = model.predict(input_df)
        return prediction[0] if hasattr(prediction, "__len__") and len(prediction) > 0 else prediction
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")