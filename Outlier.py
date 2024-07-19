import numpy as np
import matplotlib.pyplot as plt

# Your dataset as a list or NumPy array
data = [48.68,55.65,48.68,35.42,35.42,37.84,53.14,33.64,55.68,55.65,29.92,37.84,79.92,29.92,21.56,24.92,
        21.56,16.96,33.64,48.68,41.45,31.64,29.92,51.32,39.42,26.1,35.42,37.84,31.64,55.68]

# Calculate the mean and standard deviation of the data
mean = np.mean(data)
std_dev = np.std(data)

# Define a threshold (e.g., 2 times the standard deviation) for identifying outliers
threshold = 2 * std_dev
# Create empty lists to store values and outliers
values_close_to_mean = []
outliers = []

# Iterate through the data and identify values
for value in data:
    if abs(value - mean) > threshold:
        outliers.append(value)
    else:
        values_close_to_mean.append(value)

# Create a scatter plot to visualize outliers
plt.scatter(range(len(data)), data, label="Data Points")
plt.scatter(range(len(outliers)), outliers, color="red", label="Outliers")

plt.axhline(y=mean, color="green", linestyle="--", label="Mean")

plt.xlabel("Data Points")
plt.ylabel("Values")
plt.legend()
plt.title("Scatter Plot with Outliers")

plt.show()

print("Values close to the mean:", values_close_to_mean)
print("Outliers:", outliers)
# print(np.mean(values_close_to_mean))
