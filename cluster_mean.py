import numpy as np

# data = [48.68, 55.65, 48.68, 35.42, 35.42, 37.84, 53.14, 33.64, 55.68, 55.65, 29.92, 37.84, 33.64]
# a=np.mean(data)
# print(a)

Cluster1 = [48.68,55.65,48.68,53.14,55.68,55.65,48.68,51.32,55.68]
print(np.mean(Cluster1))

Cluster2 = [35.42,35.42,37.84,33.64,29.92,37.84,29.92,33.64,41.45,31.64,29.92,39.42,35.42,37.84,31.64]
print(np.mean(Cluster2))

Cluster3 = [21.56,24.92,21.56,16.96,26.1]
print(np.mean(Cluster3))
print()
print(min(Cluster1))
print(min(Cluster2))
print(max(Cluster3))
