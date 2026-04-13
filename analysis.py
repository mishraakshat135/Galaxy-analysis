#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import time
import joblib
from scipy.stats import zscore

# Machine learning libraries

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load dataset
df = pd.read_csv("star_classification.csv")

print("\nDataset loaded successfully\n")

# Data cleaning

print("\nChecking missing values:\n")
print(df.isnull().sum())
df = df.dropna()
df = df.drop_duplicates()
df.reset_index(drop=True, inplace=True)
print("\nData cleaned successfully")

# Count types of objects

type_counts = df["class"].value_counts()

print("\nObject Type Counts:\n")
print(type_counts)

# Bar chart

plt.figure()

type_counts.plot(kind="bar")
plt.title("Distribution of Object Types")
plt.xlabel("Object Type")
plt.ylabel("Count")
plt.savefig("object_type_bar_chart.png")
plt.show()

# Pie Chart

plt.figure()

type_counts.plot(kind="pie")
plt.title("Percentage of Object Types")
plt.ylabel("")
plt.savefig("object_type_pie_chart.png")
plt.show()

# Histogram of brightness (r band)

plt.figure()

plt.hist(df["r"])
plt.title("Brightness Distribution")
plt.xlabel("Brightness (r band)")
plt.ylabel("Frequency")
plt.savefig("brightness_histogram.png")
plt.show()

# Scatter plot of redshift vs brightness (r band)
sample_df = df.sample(n=2000)

plt.figure(figsize=(10, 6))

plt.scatter(
    sample_df["redshift"],
    sample_df["r"],
    s=10,
    alpha=0.4
)

plt.title("Redshift vs Brightness")
plt.xlabel("Redshift")
plt.ylabel("Brightness")
plt.show()

# Box plot of brightness by object type

plt.figure()

sns.boxplot(x="class", y="r", data=df)
plt.title("Brightness Distribution by Object Type")
plt.savefig("brightness_box_plot.png")
plt.show()


# Correation heatmap    

plt.figure(figsize=(10, 6))
selected_columns = ["u", "g", "r", "i", "z", "redshift"]
corr = df[selected_columns].corr()
sns.heatmap(corr, annot=True)
plt.title("Correlation Between Brightness and Redshift")
plt.savefig("correlation_heatmap.png")
plt.show()

# Additional statistics

print("\nAverage Brightness:", np.mean(df["r"]))
print("Maximum Brightness:", np.max(df["r"]))
print("Minimum Brightness:", np.min(df["r"]))

# Saving dataframe to a new CSV file

df.to_csv("processed_star_data.csv", index=False)

print("\nData saved successfully")


# MACHINE LEARNING SECTION


# Features and target

features = ["u", "g", "r", "i", "z", "redshift"]
X = df[features]
y = df["class"]

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Creating and training the model

total_trees = 100

model = RandomForestClassifier(
    n_estimators=1,
    warm_start=True,
    random_state=42
)

print("\nTraining model...")
start_time = time.time()
for i in tqdm(range(1, total_trees + 1), desc="Training Trees"):

    model.n_estimators = i

    model.fit(X_train, y_train)
end_time = time.time()
training_time = end_time - start_time
print("\nModel trained successfully")
print(f"Training time: {training_time:.2f} seconds")

# Predictions

y_predict = model.predict(X_test)

# Accuracy

accuracy = accuracy_score(y_test, y_predict)
print("\nModel Accuracy:", accuracy)

# Confusion matrix

cm = confusion_matrix(y_test, y_predict)
print("\nConfusion Matrix:\n")
print(cm)

plt.figure()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()

# Classification report

report = classification_report(y_test, y_predict)
print("\nClassification Report:\n")
print(report)

# Feature importance

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": features,"Importance": importances})

# Sort features by importance

feature_importance_df = feature_importance_df.sort_values(by="Importance",ascending=False)

plt.figure()

plt.bar(feature_importance_df["Feature"],feature_importance_df["Importance"])
plt.title("Feature Importance for Classification")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.savefig("feature_importance.png")
plt.show()

print("\nImportance of different features:\n")
print(feature_importance_df.sort_values(by="Importance", ascending=False))

# Prediction for new data

print("\nEnter values for a new celestial object:")

u_val = float(input("u: "))
g_val = float(input("g: "))
r_val = float(input("r: "))
i_val = float(input("i: "))
z_val = float(input("z: "))
redshift_val = float(input("redshift: "))

new_data = pd.DataFrame({
    "u": [u_val],
    "g": [g_val],
    "r": [r_val],
    "i": [i_val],
    "z": [z_val],
    "redshift": [redshift_val]
})

prediction = model.predict(new_data)
print("\nPredicted Class:", prediction[0])

# Prediction probabilities

probabilities = model.predict_proba(new_data)
print("\nPrediction Confidence:")

for class_name, prob in zip(model.classes_, probabilities[0]):
    print(f"{class_name}: {prob*100:.2f}%")

current_time = datetime.now()

# Writing results to a text file

with open("model_results.txt", "a") as file:
    file.write(f"Run Time: {current_time}\n")
    file.write(f"Training Time: {training_time:.2f} seconds\n")
    file.write(f"Model Accuracy: {accuracy:.4f}\n")
    file.write(f"Predicted Class: {prediction[0]}\n")
    file.write("Prediction Confidence:\n")
    for class_name, prob in zip(model.classes_, probabilities[0]):
        file.write(f"{class_name}: {prob*100:.2f}%\n")
print("\nResults saved to model_results.txt")

# Save model

joblib.dump(model, "trained_model.pkl")

print("\nModel saved as trained_model.pkl")