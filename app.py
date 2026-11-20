import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

# ---------------------------------------------------------------------
# LoanWise — Flask Loan Approval Predictor
# Keep decision.pkl in the same folder as this app.py file.
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "decision.pkl"

FEATURES = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file",
]

# These mappings are carried forward from the earlier app.py in this
# project. The pickle contains a plain DecisionTreeClassifier, not the
# LabelEncoder objects, so it cannot prove these category-to-number
# mappings. Verify them against the original training notebook/CSV
# before relying on predictions. They follow alphabetical LabelEncoder
# ordering for the listed category values.
ENCODINGS = {
    "person_gender": {
        "female": 0,
        "male": 1,
    },
    "person_education": {
        "Associate": 0,
        "Bachelor": 1,
        "High School": 2,
        "Master": 3,
    },
    "person_home_ownership": {
        "MORTGAGE": 0,
        "OTHER": 1,
        "OWN": 2,
        "RENT": 3,
    },
    "loan_intent": {
        "DEBTCONSOLIDATION": 0,
        "EDUCATION": 1,
        "HOMEIMPROVEMENT": 2,
        "MEDICAL": 3,
        "PERSONAL": 4,
        "VENTURE": 5,
    },
    "previous_loan_defaults_on_file": {
        "No": 0,
        "Yes": 1,
    },
}

NUMERIC_FIELDS = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Place decision.pkl beside app.py."
    )

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

# Fail early if the supplied model does not match the requested schema.
model_feature_names = getattr(model, "feature_names_in_", None)
if model_feature_names is not None:
    actual_features = list(model_feature_names)
