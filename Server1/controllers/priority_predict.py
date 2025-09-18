import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier

def load_priority_model(path: str):
    """Load the random forest model for priority prediction"""
    try:
        # First try joblib
        try:
            model_data = joblib.load(path)
            print("Loaded with joblib successfully")
            return model_data
        except:
            # Fallback to pickle
            try:
                with open(path, "rb") as f:
                    model_data = pickle.load(f)
                print("Loaded with pickle successfully")
                return model_data
            except:
                print("Both joblib and pickle loading failed")
                return None
            
    except Exception as e:
        print(f"Error loading priority model: {e}")
        return None

def prepare_priority_input(data: dict, model_data):
    """Prepare input for priority prediction"""
    # Extract model and feature names
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # Create input dictionary with default values
    input_dict = {
        'years_exp_min': 0,
        'years_exp_max': 0,
        'skills_coverage_band': 1,  # medium
        'referral_flag': 0,
        'english_level': 3,  # b1
        'location_match_local': 0,
        'location_match_relocate': 0,
        'location_match_remoteok': 0
    }
    
    # Update with provided data
    if 'years_experience' in data:
        years_exp = float(data['years_experience'])
        if years_exp <= 1:
            input_dict['years_exp_min'], input_dict['years_exp_max'] = 0, 1
        elif years_exp <= 3:
            input_dict['years_exp_min'], input_dict['years_exp_max'] = 1, 3
        elif years_exp <= 6:
            input_dict['years_exp_min'], input_dict['years_exp_max'] = 3, 6
        else:
            input_dict['years_exp_min'], input_dict['years_exp_max'] = 6, 99
    
    # Map skills coverage
    if 'skills' in data:
        skills = str(data['skills']).lower()
        if 'high' in skills:
            input_dict['skills_coverage_band'] = 2
        elif 'medium' in skills:
            input_dict['skills_coverage_band'] = 1
        else:
            input_dict['skills_coverage_band'] = 0
    
    # Referral flag
    if 'referral' in data:
        input_dict['referral_flag'] = 1 if str(data['referral']).lower() in ['true', 'yes', '1'] else 0
    
    # English level
    if 'english_level' in data:
        eng_level = str(data['english_level']).upper()
        eng_map = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}
        input_dict['english_level'] = eng_map.get(eng_level, 3)
    
    # Location match
    location = str(data.get('location', '')).lower()
    if 'local' in location:
        input_dict['location_match_local'] = 1
    elif 'relocat' in location:
        input_dict['location_match_relocate'] = 1
    elif 'remote' in location:
        input_dict['location_match_remoteok'] = 1
    else:
        # Default to local if no location specified
        input_dict['location_match_local'] = 1
    
    # Create DataFrame in correct feature order
    input_df = pd.DataFrame([input_dict])
    
    # Ensure all expected columns are present
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training order
    input_df = input_df[feature_names]
    
    return input_df, model

def predict_priority(model_path: str, data: dict):
    """Predict candidate priority"""
    model_data = load_priority_model(model_path)
    
    if model_data is None or 'model' not in model_data:
        return {"priority": "Medium", "confidence": "0%", "error": "Model not loaded properly"}
    
    try:
        input_df, model = prepare_priority_input(data, model_data)
        
        # Predict
        priority_num = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        # Map to priority labels
        priority_map = model_data.get('priority_map', {0: "Low", 1: "Medium", 2: "High"})
        priority_label = priority_map.get(priority_num, "Medium")
        
        confidence = max(probabilities) * 100
        
        return {
            "priority": priority_label,
            "confidence": f"{confidence:.1f}%",
            "probabilities": {
                "Low": f"{probabilities[0]*100:.1f}%",
                "Medium": f"{probabilities[1]*100:.1f}%",
                "High": f"{probabilities[2]*100:.1f}%"
            }
        }
        
    except Exception as e:
        print(f"Priority prediction failed: {e}")
        return {"priority": "Medium", "confidence": "0%", "error": str(e)}