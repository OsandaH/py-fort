#### K-means Clustering ####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

df = pd.read_csv('Mall_Customers.csv')

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

k_values = [2, 3, 4, 5, 6, 7, 8]

# Store silhouette scores
silhouette_scores = []
inertia_values = []

# Perform K-means clustering for different values of k
plt.figure(figsize=(18, 12))
for i, k in enumerate(k_values, 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    # Compute silhouette score
    silhouette_scores.append(silhouette_score(X, labels))
    inertia_values.append(kmeans.inertia_)
    
    plt.subplot(3, 3, i)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', edgecolors='k', s=50, alpha=0.8)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title(f'K-Means Clustering (k={k})')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    
plt.tight_layout()
plt.show()

# Find optimal k using Silhouette Score
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.show()

optimal_k = k_values[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters based on Silhouette Score is: k={optimal_k}")


# Find optimal k using Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker='o', linestyle='--', color='r')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method: Inertia vs. Number of Clusters')
plt.show()

optimal_k = k_values[np.diff(inertia_values, 2).argmin() + 1]
print(f"\nOptimal number of clusters based on Elbow Method is: k={optimal_k}")

df = pd.read_csv('Mall_Customers.csv')

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

k_values = [2, 3, 4, 5, 6, 7, 8]

# Store silhouette scores
silhouette_scores = []
log_likelihood_values = []

plt.figure(figsize=(18, 12))
for i, k in enumerate(k_values, 1):
    gmm = GaussianMixture(n_components=k, random_state=42)
    labels = gmm.fit_predict(X)
    centers = gmm.means_
    
    # Compute silhouette score
    silhouette_scores.append(silhouette_score(X, labels))
    log_likelihood_values.append(gmm.score(X) * len(X))
    
    plt.subplot(3, 3, i)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', edgecolors='k', s=50, alpha=0.8)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
    
    plt.title(f'GMM Clustering (k={k})')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    
plt.tight_layout()
plt.show()

# Find optimal k using Silhouette Score
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.show()

optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters based on Silhouette Score (GMM) is: k={optimal_k_silhouette}")


# Plot log-likelihood vs. number of clusters
plt.figure(figsize=(8, 5))
plt.plot(k_values, log_likelihood_values, marker='o', linestyle='--', color='purple')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood vs. Number of Clusters')
plt.show()

optimal_k_likelihood = k_values[np.argmax(log_likelihood_values)]
print(f"\nOptimal number of clusters based on Log-Likelihood (GMM) is: k={optimal_k_likelihood}")

# dentify the three clusters in Iris dataset using k-means clustering and use  mutual information (MI) to validate the model.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mutual_info_score
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y_true = iris.target

# number of clusters
n_clusters = 3

# K-Means Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
kmeans_mi = mutual_info_score(y_true, kmeans_labels)

print(f'K-Means Mutual Information Score: {kmeans_mi:.4f}')

# GMM Clustering
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm_labels = gmm.fit_predict(X)
gmm_mi = mutual_info_score(y_true, gmm_labels)

print(f'GMM Mutual Information Score: {gmm_mi:.4f}')



