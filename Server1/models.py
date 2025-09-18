from flask import Flask
from mongoengine import connect, Document, StringField, FloatField, ListField

app = Flask(__name__)



# Candidate model
class Candidate(Document):
    candidate_id = StringField(required=True, unique=True)
    name = StringField(required=True, max_length=100)
    current_role = StringField(max_length=100)
    years_experience = FloatField()
    skills = ListField(StringField())
    degree = StringField(max_length=50)

    def to_dict(self):
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "current_role": self.current_role,
            "years_experience": self.years_experience,
            "skills": self.skills,
            "degree": self.degree
        }

# Job model
class Job(Document):
    jd_text_128 = StringField(required=True)
    job_family = StringField(max_length=50)
    seniority = StringField(max_length=50)
    required_skills = ListField(StringField())

    def to_dict(self):
        return {
            "id": str(self.id),
            "jd_text_128": self.jd_text_128,
            "job_family": self.job_family,
            "seniority": self.seniority,
            "required_skills": self.required_skills
        }
