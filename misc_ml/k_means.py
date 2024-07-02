import numpy as np
from typing import Optional

class KMeans:
    def __init__(self, n_clusters: int, max_iter: int = 300, random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None


    def fit(self, X: np.ndarray) : 
        if self.random_state is not None:
            np.random.seed(self.random_state)

    
        # Initialize the centroids
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            self.labels = self._assign_clusters(X)

            new_centroids = self._update_centroids(X)

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return self
    
    def predict(self, X):
        return self._assign_clusters(X)
    

    def _assign_clusters(self, X):
        distances = self._calculate_distances(X, self.centroids)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X):
        return np.array([X[self.labels == i].mean(axis = 0) for i in range(self.n_clusters)])


    @staticmethod
    def _calculate_distances(X, centroids):
        """Calculate distances between each point in X and each centroid."""
        return np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
    


# Generate some random 2D data
np.random.seed(42)
X = np.random.randn(300, 2) * 0.5
X[:100, 0] += 2
X[100:200, 0] -= 2
X[200:, 1] += 2

# Create and fit KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

print("Centroids:")
print(kmeans.centroids)
print("\nFirst few labels:")
print(kmeans.labels[:10])

import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.title('K-means Clustering Results')
plt.show()