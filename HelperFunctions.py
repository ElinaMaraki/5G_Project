import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import seaborn as sns

def mbps_to_kbps(value):
    if 'Mbps' in value:
        n = float(value.replace(' Mbps',''))
        return str(n*1000)+' Kbps'
    else:
        return value

def compute_c_index(data, labels):
    
    # Compute pairwise Euclidean distances
    pairwise_distances = squareform(pdist(data, metric='euclidean'))

    # Collect distances within clusters
    S = []
    all_distances = []

    for i in range(len(data)):
        for j in range(i + 1, len(data)):  # Avoid double-counting pairs
            if labels[i] == labels[j]:  # Same cluster
                S.append(pairwise_distances[i, j])
            all_distances.append(pairwise_distances[i, j])

    S = np.array(S)
    all_distances = np.array(all_distances)

    # Compute S_min and S_max (smallest and largest possible sums)
    S_min = np.sum(np.sort(all_distances)[:len(S)])
    S_max = np.sum(np.sort(all_distances)[-len(S):])

    # Compute C-Index
    C_index = (np.sum(S) - S_min) / (S_max - S_min)
    return C_index


def plot_heatmap(correlation_matrix):
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title("Heatmap of QoS Features")
    plt.show()

def plot_elbow_kmeans(K_range, wcss):
    # Plot WSS - num of Clusters to find Elbow
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, wcss, marker='o', linestyle='-')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.title('Elbow Method for K-Means')
    plt.show()

def plot_dendogram(linkage_matrix):
    # Plot dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram = sch.dendrogram(linkage_matrix)
    plt.title("Dendrogram")
    plt.xlabel("Data Points")
    plt.ylabel("Euclidean Distance")
    plt.show()

def plot_elbow_ward(num_clusters, distances):
    plt.figure(figsize=(8, 5))
    plt.plot(num_clusters, distances, marker='o', linestyle='-')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Ward Linkage Distance")
    plt.title("Elbow Method for Ward Linkage")
    plt.show()
