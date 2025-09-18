import requests
from bson.json_util import dumps
from models import Candidate, Job
from controllers.salary_predet import predict



modalPath="../models/random_forest.pkl"



# Fetch a single candidate
def get_candidate(candidate_id: str):

    try:
        # Call external API
        try:
            api_response = requests.get(f"http://localhost:5000/api/candidate/{candidate_id}")
            if api_response.status_code == 200:
               return  api_response.json()
        except Exception as api_err:
            return "erro"

        return api_response.json()

    except Exception as e:
        return {"error": f"Failed to retrieve candidate: {str(e)}"}


# Fetch a single job
def get_job(job_id: str):
    try:
        # Call external API
        api_response = requests.get(f"http://localhost:5000/api/job/{job_id}")
        if api_response.status_code == 200:
            try:
                return api_response.json()
            except ValueError:
                return {"error": "Invalid JSON from job API"}
        if api_response.status_code == 404:
            return {"error": "Job not found", "status": 404}
        return {"error": "Job API error", "status": api_response.status_code, "text": api_response.text}
    except Exception as api_err:
        return {"error": f"Failed to retrieve job: {str(api_err)}"}


# List all candidates
def list_candidates():
    try:
       

        # Call external API
        try:
            api_response = requests.get("http://localhost:5000/api/candidates")
            if api_response.status_code == 200:
                return api_response.json()
        except Exception as api_err:
            print("API call error:", api_err)

        return dumps(api_response.json())

    except Exception as e:
        return {"error": f"Failed to retrieve candidates: {str(e)}"}


# List all jobs
def list_jobs():
    try:

        try:
            api_response = requests.get("http://localhost:5000/api/jobs")
            if api_response.status_code == 200:
               return (api_response.json())
        except Exception as api_err:
            print("API call error:", api_err)

        return dumps(api_response.json())

    except Exception as e:
        return {"error": f"Failed to retrieve jobs: {str(e)}"}


def check_job_fit(candidate_id: str, job_id: str):
    """
    Performs an ML-powered analysis to check a candidate's fit for a job.
    This function would call your job-fit model.
    """
    # This is a placeholder; you'd integrate your actual ML model here.
    # Example: score = my_ml_model.predict_fit(candidate_data, job_data)
    return f"Candidate {candidate_id} is a good fit for job {job_id} with an 85% score."

def predict_salary(data: dict):
    """
    Predicts the salary for a given role and years of experience.

    Args:
        data (dict): Example format:
            {
                "years_exp_band": 5,
                "english_level": 0,
                "referral_flag": 1,
                "location_match": "Location match local",
                "skills_coverage_band": 0,
            }

    Returns:
        str: predicted salary like "$85,000.00"
        or dict: {"error": "..."} if inputs are invalid
    """
    try:
        # Validate required keys
        required = [
            "years_exp_band",
            "english_level",
            "referral_flag",
            "location_match",
            "skills_coverage_band",
        ]
        for key in required:
            if key not in data:
                return {"error": f"Missing required field: {key}"}

        # Run prediction using your predictor
        prediction = predict(modalPath,data)

        # Format salary (if it's numeric)
        if isinstance(prediction, (int, float)):
            return f"${prediction:,.2f}"
        return str(prediction)

    except Exception as exc:
        return {"error": str(exc)}




def predict_salary_tool(data: dict):
    """
    Tool-friendly wrapper for predicting salary. Expects a dict input like:
      {"role": "Data Scientist", "years_experience": 3}

    Returns a dict with either {"prediction": "$85,000.00"}
    or {"error": ..., "hint": ..., "usage": ...} when invalid.
    """
    role = None
    years_experience = None

    if isinstance(data, dict):
        role = data.get("role")
        years_experience = data.get("years_experience")

    result = predict_salary(role, years_experience)
    if isinstance(result, dict) and result.get("error"):
        return result

    return {"prediction": result}

def screen_resume(candidate_id: str):
    """
    Performs an automated resume screening for a candidate.
    """
    # Placeholder for a resume screening model.
    return f"Resume for candidate {candidate_id} has been screened and recommended for the next round."

def get_priority(candidate_id: str):
    """
    Retrieves the priority level for a candidate based on internal scoring.
    """
    # Placeholder for a candidate priority function.
    priority = "High" if candidate_id == "CAND001" else "Medium"
    return f"Candidate {candidate_id} has a {priority} priority."
