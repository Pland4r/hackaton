from flask import Blueprint, request, jsonify
from controllers.salary_predet import predict
import pickle
import pandas as pd
from controllers.priority_predict import predict_priority
from controllers.job_fit_predict import predict_job_fit

# Create a blueprint
ml_bp = Blueprint("ml", __name__)
PredectPath="D:/hackathon-main/hackathon-main/server/models/linear_regression.pkl"

# Path to the random forest model
RANDOM_FOREST_PATH = "D:/hackathon-main/hackathon-main/server/models/random_forest.pkl"

JOB_FIT_MODEL_PATH = "D:/hackathon-main/hackathon-main/server/models/logistical_regression.pkl"

@ml_bp.route("/salary/predict", methods=["POST"])
def predict_salary():
    """Predict salary based on candidate/job features."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        print("Received data:", data)
        
        # Validate required fields
        required_fields = ['role', 'years_experience', 'degree', 'company_size', 'location', 'level']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Convert years_experience to float
        try:
            data['years_experience'] = float(data['years_experience'])
        except ValueError:
            return jsonify({"error": "years_experience must be a number"}), 400
        
        print("Starting prediction with data:", data)
        
        # Make prediction
        prediction = predict(PredectPath, data)
        
        # Format the prediction result
        if isinstance(prediction, (int, float)):
            return jsonify({"salary": f"${prediction:,.2f}"})
        return jsonify({"salary": str(prediction)})

    except Exception as exc:
        return jsonify({"error": f"Prediction error: {str(exc)}"}), 500


@ml_bp.route("/job_fit/predict", methods=["POST"])
def predict_job_fit_route():  # ← Rename this function
    """Predict job fit score using logistic regression model."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        print("Received job fit prediction request:", data)
        
        # Validate required fields
        required_fields = ['required_skills', 'candidate_skills', 'degree', 'years_experience']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Make prediction using the logistic regression controller
        result = predict_job_fit(JOB_FIT_MODEL_PATH, data)  # This now calls the controller function
        
        response = {
            "fit_score": result.get('fit_score'),
            "probability": result.get('probability'),
            "is_good_fit": result.get('is_good_fit'),
            "prediction": result.get('prediction')
        }
        
        return jsonify(response)
        
    except FileNotFoundError as exc:
        return jsonify({"error": f"Model not found: {str(exc)}"}), 500
    except ValueError as exc:
        return jsonify({"error": f"Invalid model data: {str(exc)}"}), 500
    except RuntimeError as exc:
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500
    except Exception as exc:
        return jsonify({"error": f"Job fit prediction failed: {str(exc)}"}), 500
        
@ml_bp.route("/resume_screen/predict", methods=["POST"])
def resume_screen():
    """Screen candidate resume (dummy example)."""
    try:
        data = request.get_json()
        if not data or "candidate_id" not in data:
            return jsonify({"error": "candidate_id required"}), 400

        # Dummy response
        result = f"Candidate {data['candidate_id']} resume looks strong."

        return jsonify({"screening_result": result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@ml_bp.route("/candidate_priority/predict", methods=["POST"])
def candidate_priority():
    """Predict candidate priority using the random forest model."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Make prediction
        result = predict_priority(RANDOM_FOREST_PATH, data)
        
        # If there was an error, return it
        if "error" in result:
            return jsonify({"error": result["error"]}), 500
        
        return jsonify(result)
        
    except Exception as exc:
        return jsonify({"error": f"Priority prediction failed: {str(exc)}"}), 500