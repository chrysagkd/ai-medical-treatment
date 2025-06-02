# therapy_hypertension_simple.py

def recommend_treatment(systolic_bp, diastolic_bp):
    """
    Simple rule-based treatment recommendation for hypertension.
    Parameters:
      systolic_bp (int): Systolic blood pressure
      diastolic_bp (int): Diastolic blood pressure
    Returns:
      str: Recommended treatment
    """

    if systolic_bp < 120 and diastolic_bp < 80:
        return "No treatment needed. Maintain healthy lifestyle."
    elif 120 <= systolic_bp < 130 and diastolic_bp < 80:
        return "Recommend lifestyle modifications (diet, exercise)."
    elif (130 <= systolic_bp < 140) or (80 <= diastolic_bp < 90):
        return "Consider starting antihypertensive medication if risk factors present."
    elif systolic_bp >= 140 or diastolic_bp >= 90:
        return "Start antihypertensive medication. Monitor regularly."
    else:
        return "Consult your healthcare provider for further evaluation."

# --- Demo ---
if __name__ == "__main__":
    patients = [
        {"id": 1, "systolic": 118, "diastolic": 76},
        {"id": 2, "systolic": 125, "diastolic": 79},
        {"id": 3, "systolic": 135, "diastolic": 85},
        {"id": 4, "systolic": 145, "diastolic": 95},
    ]

    for p in patients:
        treatment = recommend_treatment(p['systolic'], p['diastolic'])
        print(f"Patient {p['id']}: SBP={p['systolic']}, DBP={p['diastolic']} -> {treatment}")
