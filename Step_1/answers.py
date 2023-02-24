# https://machinelearningmastery.ru/a-complete-guide-to-principal-component-analysis-pca-in-machine-learning-664f34fc3e5a/

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

data = np.genfromtxt('14_16.csv', delimiter=',')
print(data.shape)


# Convert data to a NumPy array
#X = data.values
X = data
# Perform PCA with two components
pca = PCA(n_components=2, svd_solver='full')
pca.fit(X)
X_pca = pca.transform(X)

# Calculate the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print(explained_variance_ratio)

# Find the first object's coordinates with respect to the first two principal components
first_object_coords = X_pca[0]
print(f"First object's coordinates with respect to the first two principal components: {first_object_coords}")

# Calculate the total explained variance for the first two components
total_variance = np.sum(explained_variance_ratio[:2])

# Print the total explained variance for the first two components
print(f"Total explained variance for the first two components: {total_variance}")

# Find the first object's coordinate with respect to the first principal component
first_object_coord = X_pca[0, 0]
print(f"First object's coordinate with respect to the first principal component: {first_object_coord}")

