import pandas as pd
import joblib
import os

print("\nLoading trained model...")
model = joblib.load("trained_model.pkl")
print("Model loaded successfully")
input_file = "scientist_input.csv"

if os.path.exists(input_file):

    print("\nReading scientist input data...")
    data = pd.read_csv(input_file)
    print("Number of observations:", len(data))

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
