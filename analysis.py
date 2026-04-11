#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# -----------------------------
# MACHINE LEARNING SECTION
# -----------------------------

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

# Create model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model

model.fit(X_train, y_train)

print("\nModel trained successfully")

# Predictions

y_pred = model.predict(X_test)

# Accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Confusion matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")

print(cm)

import seaborn as sns
import matplotlib.pyplot as plt

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

report = classification_report(y_test, y_pred)

print("\nClassification Report:\n")

print(report)

# Feature importance

importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
})

print("\nFeature Importance:\n")

print(feature_importance_df.sort_values(by="Importance", ascending=False))