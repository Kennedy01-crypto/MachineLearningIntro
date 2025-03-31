# import the neccesary libraries
import pandas as pd
from sklearn.datasets import load_iris

# load the dataset
iris = load_iris()

# feature selection
X = iris.data
y = iris.target

feature_names = X.feature_names
target_names = y.target_names

# data preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# building and training the model
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state= 42)
kmeans.fir(X_scaled)

# Visualize Clustering pooints
import matplotlib.pyplot as plt
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.7)

plt.scatter(kmeans.cluster_centers_ [:, 0], kmeans.cluster_centers_[:, 1], s=200, marker = 'X', c='red', label='Centers')
plt.xlabel("Feature 1(Standardized)")
plt.ylabel("Feature 2(Standardized)")
plt.legend()
plt.title("K means Clustering on Iris Dataset")
plt.show()