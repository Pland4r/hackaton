import torch
import torch.nn as nn
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

# ----------------------------
# ⿡ Device setup
# ----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ----------------------------
# ⿢ Load encoders
# ----------------------------
ohe_job = joblib.load("ohe_job.pkl")
ohe_seniority = joblib.load("ohe_seniority.pkl")

# ----------------------------
# ⿣ Load SBERT model
# ----------------------------
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# ----------------------------
# ⿤ Neural network class
# ----------------------------
class AdvancePredictor(nn.Module):
    def _init_(self, resume_dim, jd_dim, job_dim, seniority_dim, numeric_dim):
        super()._init_()
        input_dim = resume_dim + jd_dim + job_dim + seniority_dim + numeric_dim + 1  # +1 for cosine sim
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, resume, jd, job, seniority, numeric):
        cos_sim = torch.sum(resume * jd, dim=1, keepdim=True) / (
            torch.norm(resume, dim=1, keepdim=True) * torch.norm(jd, dim=1, keepdim=True) + 1e-6
        )
        x = torch.cat([resume, jd, job, seniority, numeric, cos_sim], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.out(x))
        return x

# ----------------------------
# ⿥ Load trained model
# ----------------------------
resume_dim = 384
jd_dim = 384
job_dim = ohe_job.transform(pd.DataFrame({"job_family": ["Backend"]})).shape[1]
seniority_dim = ohe_seniority.transform(pd.DataFrame({"seniority": ["Mid"]})).shape[1]
numeric_dim = 2  # years_experience + skill_overlap

model = AdvancePredictor(resume_dim, jd_dim, job_dim, seniority_dim, numeric_dim)
model.load_state_dict(torch.load("advance_predictor_model.pt", map_location=device))
model.to(device)
model.eval()

# ----------------------------
# ⿦ Feature engineering functions
# ----------------------------
import re

skills_list = ['Python', 'AWS', 'DevOps', 'QA', 'Frontend', 'Backend', 'PM']

def extract_years(resume_text):
    match = re.search(r'(\d+)\+?\s*years', resume_text)
    return int(match.group(1)) if match else 0

def skill_overlap(resume_text, jd_text):
    resume_skills = set([s for s in skills_list if s.lower() in resume_text.lower()])
    jd_skills = set([s for s in skills_list if s.lower() in jd_text.lower()])
    return len(resume_skills & jd_skills) / (len(jd_skills) + 1e-6)

# ----------------------------
# ⿧ Prediction function
# ----------------------------
def predict_advance(resume_text, jd_text, job_family, seniority):
    # Text embeddings
    resume_emb = torch.tensor(sbert_model.encode([resume_text], convert_to_numpy=True), dtype=torch.float32).to(device)
    jd_emb = torch.tensor(sbert_model.encode([jd_text], convert_to_numpy=True), dtype=torch.float32).to(device)

    # Categorical features (use DataFrame to avoid warnings)
    job_enc = torch.tensor(ohe_job.transform(pd.DataFrame({"job_family": [job_family]})), dtype=torch.float32).to(device)
    seniority_enc = torch.tensor(ohe_seniority.transform(pd.DataFrame({"seniority": [seniority]})), dtype=torch.float32).to(device)

    # Numeric features
    years_exp = extract_years(resume_text)
    overlap = skill_overlap(resume_text, jd_text)
    numeric_feats = torch.tensor([[years_exp, overlap]], dtype=torch.float32).to(device)

    # Forward pass
    with torch.no_grad():
        prob = model(resume_emb, jd_emb, job_enc, seniority_enc, numeric_feats)
    return float(prob.item())

# ----------------------------
# ⿨ Example usage
# ----------------------------
# Pick a real datapoint from your dataset




sample = df.iloc[0]  # first row
resume_text = sample['resume_text_256']
jd_text = sample['jd_text_128']
job_family = sample['job_family']
seniority = sample['seniority']

# Predict
pred = predict_advance(resume_text, jd_text, job_family, seniority)
print(f"Predicted Advance Probability: {pred:.4f}")