import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Define your data
data = [48.68, 55.65, 48.68, 35.42, 35.42, 37.84, 53.14, 33.64, 55.68, 55.65, 29.92, 37.84, 29.92, 21.56, 24.92,
        21.56, 16.96, 33.64, 48.68, 41.45, 31.64, 29.92, 51.32, 39.42, 26.1, 35.42, 37.84, 31.64, 55.68]

# Reshape the data for clustering (required for scikit-learn)
data = np.array(data).reshape(-1, 1)

# Number of clusters
num_clusters = 3  # You modified this to 3 clusters

# Create a K-Means model
kmeans = KMeans(n_clusters=num_clusters)

# Fit the model to the data
kmeans.fit(data)

# Get cluster assignments for each data point
cluster_assignments = kmeans.labels_

# Get cluster centers
cluster_centers = kmeans.cluster_centers_

# Visualize the clusters
for i in range(num_clusters):
    plt.scatter(data[cluster_assignments == i], [0] * np.sum(cluster_assignments == i), label=f'Cluster {i + 1}')

# Plot cluster centers
plt.scatter(cluster_centers, [0] * num_clusters, s=100, c='red', label='Cluster Centers')

plt.title('Data Clustering')
plt.legend()
plt.show()

# Display data points for each cluster
for i in range(num_clusters):
    cluster_data = data[cluster_assignments == i]
    print(f"Cluster {i + 1} Data:")
    print(cluster_data)
