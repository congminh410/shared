import numpy as np
import matplotlib.pyplot as plt


def dbscan(data, eps, min_samples):
    labels = np.zeros(len(data))
    cluster_id = 0

    for i in range(len(data)):
        if labels[i] != 0:
            continue

        neighbors = get_neighbors(data, i, eps)

        if len(neighbors) < min_samples:
            labels[i] = -1  # Mark as noise
        else:
            cluster_id += 1
            expand_cluster(data, labels, i, neighbors, cluster_id, eps, min_samples)

    return labels

def get_neighbors(data, query_point_index, eps):
    '''
    Detect eps-neighborhood of query point
    
    Return all points have euclidean distance from query point < eps
    '''
    neighbors = []
    for i in range(len(data)):
        if np.linalg.norm(data[query_point_index] - data[i]) < eps:  # Euclidean distance < eps
            neighbors.append(i)
    return neighbors

def expand_cluster(data, labels, core_point_index, neighbors, cluster_id, eps, min_samples):
    labels[core_point_index] = cluster_id

    for neighbor in neighbors:
        if labels[neighbor] == -1:
            labels[neighbor] = cluster_id  # Change noise to border point
        elif labels[neighbor] == 0:
            labels[neighbor] = cluster_id
            new_neighbors = get_neighbors(data, neighbor, eps)
            
            if len(new_neighbors) >= min_samples:
                neighbors.extend(new_neighbors)

# Generate sample data
data = np.concatenate([np.random.normal(loc=0, scale=.5, size=(100, 2)),
                       np.random.normal(loc=4, scale=.5, size=(100, 2)),
                       np.random.normal(loc=8, scale=.5, size=(100, 2))])

# DBSCAN parameters
eps = 1
min_samples = 5

# Run DBSCAN
labels = dbscan(data, eps, min_samples)

# Visualize the results
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolors='k')
plt.title('DBSCAN Clustering')
plt.show()
