#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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