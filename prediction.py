import pandas as pd
import joblib
import os
from scipy.stats import zscore

print("\nLoading trained model...")
model = joblib.load("trained_model.pkl")
print("Model loaded successfully")
input_file = "scientist_input.csv"

if os.path.exists(input_file):

    print("\nReading scientist input data...")
    data = pd.read_csv(input_file)
    print("Number of observations:", len(data))

    # ==============================
# OUTLIER DETECTION IN PREDICTION
# ==============================



    print("\nChecking for rare observations in input data...")

    columns_to_check = ["u", "g", "r", "i", "z", "redshift"]

    # Calculate Z-scores

    z_scores = data[columns_to_check].apply(zscore)

    # Identify rare rows

    rare_rows = data[(z_scores.abs() > 3).any(axis=1)]

    print("Number of rare observations detected:", len(rare_rows))

    # Save rare input data

    if len(rare_rows) > 0:

        rare_rows.to_csv(
            "rare_inputs_detected.csv",
            index=False
        )

        print("Rare input data saved to rare_inputs_detected.csv")

    else:

        print("No rare observations detected.")

    # Predictions

    predictions = model.predict(data)
    probabilities = model.predict_proba(data)
    data["predicted_class"] = predictions

    for i, class_name in enumerate(model.classes_):
        data[f"prob_{class_name}"] = probabilities[:, i]

    output_file = "scientist_predictions.csv"
    data.to_csv(output_file, index=False)
    print("\nPredictions completed")
    print("Results saved to:", output_file)

else:

    print("\nscientist_input.csv file not found")
