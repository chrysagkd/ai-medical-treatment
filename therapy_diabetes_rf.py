# therapy_diabetes_rf.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Mock dataset ---
# Χρησιμοποιούμε συνθετικά δεδομένα για απλότητα
np.random.seed(42)
n_samples = 200

data = pd.DataFrame({
    'age': np.random.randint(30, 80, n_samples),
    'bmi': np.random.uniform(18, 40, n_samples),
    'hba1c': np.random.uniform(5.0, 10.0, n_samples),
    'duration_diabetes': np.random.randint(0, 20, n_samples),
    'has_complications': np.random.choice([0, 1], n_samples),
    # target: 0 = lifestyle only, 1 = oral meds, 2 = insulin therapy
    'treatment': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2])
})

# --- Features & Target ---
X = data.drop(columns='treatment')
y = data['treatment']

# --- Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- Model ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# --- Evaluation ---
y_pred = rf.predict(X_test)
print("Classification Report for Diabetes Treatment Prediction:\n")
print(classification_report(y_test, y_pred, target_names=['Lifestyle', 'Oral Meds', 'Insulin']))

# --- Function for new patient prediction ---
def predict_treatment(age, bmi, hba1c, duration, complications):
    features = np.array([[age, bmi, hba1c, duration, complications]])
    pred = rf.predict(features)[0]
    treatments = {0: 'Lifestyle Modification', 1: 'Oral Medications', 2: 'Insulin Therapy'}
    return treatments.get(pred, "Unknown")

# --- Demo ---
if __name__ == "__main__":
    new_patient = {'age': 55, 'bmi': 28.4, 'hba1c': 7.8, 'duration': 5, 'complications': 1}
    recommendation = predict_treatment(**new_patient)
    print(f"Recommended treatment for new patient: {recommendation}")
