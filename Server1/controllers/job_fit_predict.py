import pandas as pd
import pickle
import numpy as np
from typing import Dict, Any, Optional, Union

def load_job_fit_model(path: str) -> Optional[Dict[str, Any]]:
    """
    Load the job fit prediction model and its preprocessing components.
    """
    try:
        with open(path, "rb") as f:
            model_data = pickle.load(f)
            
        print(f"Loaded job fit model object type: {type(model_data)}")
        
        # Handle case where it's just the raw LogisticRegression object
        if hasattr(model_data, 'predict') and hasattr(model_data, 'predict_proba'):
            print("✅ Found raw LogisticRegression model object")
            print(f"Model parameters: {model_data.get_params()}")
            
            # Create a dictionary format with the model
            return {
                'model': model_data,
                'scaler': None,  # No scaler included
                'encoder': None,  # No encoder included
                'ordinal_maps': {},  # No ordinal mappings
                'rare_categories': {},  # No rare categories handling
                'model_columns': []  # You'll need to define expected columns
            }
            
        # Handle case where it's a dictionary with model and preprocessing
        elif isinstance(model_data, dict) and 'model' in model_data:
            print("✅ Found job fit model in dictionary format")
            return model_data
            
        else:
            print("❌ Model format not recognized for job fit prediction")
            return None
            
    except FileNotFoundError:
        print(f"❌ Job fit model file not found at: {path}")
        return None
    except Exception as e:
        print(f"⚠️ Error loading job fit model: {e}")
        return None

def get_expected_features_from_model(model):
    """
    Try to infer expected features from the model.
    This is a best-effort approach for raw models.
    """
    # If it's a logistic regression model with coef_, we can infer feature count
    if hasattr(model, 'coef_'):
        num_features = model.coef_.shape[1] if len(model.coef_.shape) > 1 else len(model.coef_)
        print(f"Model expects {num_features} features")
        return [f"feature_{i}" for i in range(num_features)]
    
    # For other models, we might need different approaches
    return []

def prepare_job_fit_input(data: Dict[str, Any], model_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Prepare input data for job fit prediction.
    """
    model = model_data['model']
    
    # If we don't have predefined model_columns, try to infer them
    if not model_data.get('model_columns'):
        model_data['model_columns'] = get_expected_features_from_model(model)
    
    # Create DataFrame from input
    input_df = pd.DataFrame([data])
    
    print(f"Raw input features: {list(input_df.columns)}")
    print(f"Expected features: {model_data.get('model_columns', [])}")
    
    # For raw models without preprocessing, we need to handle feature engineering
    if (model_data.get('scaler') is None and 
        model_data.get('encoder') is None and 
        not model_data.get('ordinal_maps') and 
        not model_data.get('rare_categories')):
        
        print("⚠️ No preprocessing components found. Performing basic feature preparation.")
        
        # Convert all features to numeric
        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                # Simple encoding for categorical variables
                input_df[col] = pd.factorize(input_df[col])[0]
        
        # Ensure numeric types and handle missing values
        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # If we know the expected number of features, ensure we have the right count
        expected_columns = model_data.get('model_columns', [])
        if expected_columns:
            # Add missing columns with default value 0
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns to match expected order
            input_df = input_df.reindex(columns=expected_columns, fill_value=0)
        
        print(f"Prepared input shape: {input_df.shape}")
        return input_df
    
    # ... (keep your original preprocessing code for dictionary format models) ...

def predict_job_fit(model_path: str, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict job fit score for a candidate using logistic regression model.
    
    Args:
        model_path (str): Path to the trained model pickle file
        candidate_data (Dict): Dictionary containing candidate and job features
        
    Returns:
        Dict: Prediction results including probability and fit score
    """
    # Load model data
    model_data = load_job_fit_model(model_path)
    if model_data is None:
        raise FileNotFoundError(f"Job fit model not found at {model_path}")
    
    # Check if we have a model in the dictionary
    if 'model' not in model_data:
        raise ValueError("No model found in the loaded data for job fit prediction")
    
    model = model_data['model']
    
    # Prepare input data
    try:
        input_df = prepare_job_fit_input(candidate_data, model_data)
    except Exception as e:
        raise RuntimeError(f"Input preparation failed: {e}")
    
    try:
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_df)[0]
            prediction = model.predict(input_df)[0]
            
            # For binary classification, get the positive class probability
            if len(probabilities) == 2:
                fit_probability = float(probabilities[1])  # Probability of good fit
            else:
                fit_probability = float(np.max(probabilities))
                
            result = {
                'prediction': int(prediction),
                'probability': float(fit_probability),
                'fit_score': float(fit_probability * 100),  # Convert to percentage
                'is_good_fit': bool(prediction == 1) if hasattr(prediction, 'item') else bool(prediction)
            }
        else:
            # Fallback for models without predict_proba
            prediction = model.predict(input_df)[0]
            result = {
                'prediction': int(prediction),
                'probability': None,
                'fit_score': None,
                'is_good_fit': bool(prediction == 1) if hasattr(prediction, 'item') else bool(prediction)
            }
        
        print(f"Prediction result: {result}")
        return result
        
    except Exception as e:
        raise RuntimeError(f"Job fit prediction failed: {e}")

# Example usage
if __name__ == "__main__":
    # Example candidate data (adjust based on your actual features)
    example_candidate = {
        'years_experience': 5,
        'education_level': 'Bachelor',
        'skill_match_score': 0.85,
        'salary_expectation': 75000,
        'location_match': True,
        'industry_experience': 'Technology',
        'required_skills': 8,
        'matching_skills': 7
    }
    
    try:
        # Replace with your actual model path
        model_path = "job_fit_model.pkl"
        result = predict_job_fit(model_path, example_candidate)
        print(f"Job Fit Prediction: {result}")
    except Exception as e:
        print(f"Error: {e}")