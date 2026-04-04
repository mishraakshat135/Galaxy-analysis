#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# To load dataset
df = pd.read_csv("galaxy_data.csv")

print("Dataset loaded successfully\n")

print(df.head())

# Some information
print("\nDataset Information:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())

# To count no. of each  galaxy type
print("\nNumebr of each type of galaxies:\n")
type = df["Galaxy_Type"].value_counts()
print(type)

# Calculate avg brightness
avg_brightness = df["Brightness"].mean()
print("\nAverage Brightness:", avg_brightness)

# Galaxy type bar chart
plt.figure()

type.plot(kind="bar")
plt.title("Distribution of Galaxy Types")
plt.xlabel("Galaxy Type")
plt.ylabel("Count")
plt.savefig("galaxy_type_bar_chart.png")
plt.show()

# Brightness dist. histogram
plt.figure()

plt.hist(df["Brightness"])

plt.title("Brightness Distribution")
plt.xlabel("Brightness")
plt.ylabel("Frequency")
plt.savefig("brightness_histogram.png")
plt.show()

# Redshift vs size plot
plt.figure()

plt.scatter(df["Redshift"], df["Size"])

plt.title("Redshift vs Size")
plt.xlabel("Redshift")
plt.ylabel("Size")
plt.savefig("redshift_plot.png")
plt.show()

# % of galaxy type pie chart
plt.figure()

type.plot(kind="pie", autopct="%1.1f%%") ##############

plt.title("Percentage of Galaxy Types")


plt.savefig("pie_chart.png")
plt.show()

# Brightness by galaxy type
plt.figure()

sns.boxplot(x="Galaxy_Type", y="Brightness", data=df)

plt.title("Brightness Distribution by Galaxy Type")
plt.savefig("box_plot.png")
plt.show()

# Correlation heatmap
plt.figure()

corr = df.corr(numeric_only=True)

sns.heatmap(corr, annot=True)

plt.title("Correlation Between Galaxy Properties")
plt.savefig("heatmap.png")
plt.show()