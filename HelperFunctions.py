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

    '''
    Compute pairwise Euclidean distances
    Compute distances of all datapoints with pdist which returns a 1-D array only for the unique pairs
    Computes the distance between m points using Euclidean distance (2-norm) as the distance metric between the points.
    The points are arranged as m n-dimensional row vectors in the matrix X.
    squareform converts between condensed distance matrices and square distance matrices.
    Create 2-D array for easy indexing.
    '''
    pairwise_distances = squareform(pdist(data, metric='euclidean'))

    S = []
    all_distances = []

    # Extract distances that are in the same cluster
    # To avoid double-counting pairs we use only the upper triangular values
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if labels[i] == labels[j]:
                S.append(pairwise_distances[i, j])
            all_distances.append(pairwise_distances[i, j])

    '''
    Convert S (intra-cluster  distances) and all_distances arrays to np arrays
    '''
    S = np.array(S)
    all_distances = np.array(all_distances)

    if len(S) == 0:
        return 1

    '''
    Compute S_min and S_max : 
    max and min distances between all datapoints regardless(no matter within-cluster or between-cluster a distance is)
    '''
    S_min = np.sum(np.sort(all_distances)[:len(S)])
    S_max = np.sum(np.sort(all_distances)[-len(S):])

    if S_max == S_min:
        return 0

    C_index = (np.sum(S) - S_min) / (S_max - S_min)
    return C_index

''' PLOT FUNCTIONS '''

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
