# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance


def DB_index2D(centroids, clusters, X, metric='euclidean'):
    """
    Return the Davies-Bouldin index for cluster evaluation for 2D data.

    The index is calculated using the formula:
    DB = (1/k) * sum(max((sigma_i + sigma_j)/distance(c_i, c_j)))
    where:
        k is the number of clusters
        sigma_i is the average distance of all elements in cluster i to
            centroid c_i
        c_i is the centroid of cluster i
        distance(c_i, c_j) is the distance between cluster centroids i and
            j, using the same metric as in the algorithm

    The smaller the value of the DB index, the better the clustering.

    Parameters
    ----------
    centroids : ndarray of float
        The centroids of the clusters
    clusters : dict
        Cluster id mapped to a list of the cluster's elements
    X : ndarray (2D)
        The data array
    metric : str or callable, optional
        The distance metric to use.  If a string, the distance function can be
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
        'wminkowski', 'yule'.

    Returns
    -------
    DB_ind : float
        The Davies-Bouldin index
    """
    K = len(clusters.keys())
    mean_distances = np.vstack(distance.cdist(centroids[x, np.newaxis],
                                              X.take(clusters[x], axis=0),
                                              metric=metric).mean()
                               for x in clusters.keys())
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i+1, K):
            D[i, j] = (mean_distances[i] + mean_distances[j]) / distance.cdist(centroids[i, np.newaxis], centroids[j, np.newaxis], metric=metric)

    return (1/K) * np.sum(D.max(axis=1))

def DB_index3D(centroids, clusters, X, metric='euclidean', mode=1):
    """
    Return the Davies-Bouldin index for cluster evaluation for 3D data.

    The index is calculated using the formula:
    DB = (1/k) * sum(max((sigma_i + sigma_j)/distance(c_i, c_j)))
    where:
        k is the number of clusters
        sigma_i is the average distance of all elements in cluster i to
            centroid c_i
        c_i is the centroid of cluster i
        distance(c_i, c_j) is the distance between cluster centroids i and
            j, using the same metric as in the algorithm

    The smaller the value of the DB index, the better the clustering.

    Parameters
    ----------
    centroids : ndarray of float
        The centroids of the clusters
    clusters : dict
        Cluster id mapped to a list of the cluster's elements
    X : ndarray (3D)
        The data array
    metric : str or callable, optional
        The distance metric to use.  If a string, the distance function can be
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
        'wminkowski', 'yule'.
    mode : {0, 1, 2}, default 1
        Determines the axis along which to do the clustering.
        mode=i means that the clustering will be done along axis i.

    Returns
    -------
    DB_ind : float
        The Davies-Bouldin index
    """
    centroids = np.nan_to_num(centroids)
    K = len(clusters.keys())

    mean_distances = []
    non_empty_clusters = []

    for i in sorted(clusters.keys()):
        # Ignore current cluster if empty
        if (len(clusters[i]) == 0):
            mean_distances.append(0)
            continue

        non_empty_clusters.append(i)
        # Choose only the elements of cluster i
        i_data = np.nan_to_num(X.take(clusters[i], axis=mode))

        if (mode == 0) or (mode == 1):
            # Convert i_data to 2-D array by flattening the chromosomes
            # dimension
            i_data = np.reshape(i_data, (i_data.shape[0]*i_data.shape[1],
                                         i_data.shape[2]))
            # Remove the NaN rows
            i_data = i_data.take(np.unique(np.nonzero(i_data)[0]), axis=0)
        else:
            # Take the mean of each sample for all chromosomes
            i_data = np.nan_to_num(np.nanmean(i_data, axis=0))

            i_data = np.swapaxes(i_data, 0, 1)

        # Compute the mean distances of each cluster element to the
        # cluster's centroid
        mean_distances.append(distance.cdist(centroids[i, np.newaxis],
                                             i_data).mean())

    # Compute the distortion of each cluster
    D = np.zeros((K, K))
    for i in range(K):
        if (i not in non_empty_clusters):
            continue
        for j in range(i+1, K):
            if (j not in non_empty_clusters):
                continue
            D[i, j] = (mean_distances[i] + mean_distances[j]) / distance.cdist(centroids[i, np.newaxis], centroids[j, np.newaxis])

    return (1/len(non_empty_clusters)) * np.sum(D.max(axis=1))

def modified_Gamma_index2D(centroids, clusters, labels, X, metric='euclidean'):
    """
    Return the Modified Hubert Gamma statistic.
    
    This statistic is calculated using the formula:
    Gamma = (1/M) * sum(sum(P(i,j)*Q(i,j)))
    where:
        M is the number of all possible pairs of X datapoints
        P is the similarity (distance) matrix of X datapoints
        Q is the matrix whose element Q(i,j) holds the distances between the
        centroids of the clusters that xi and xj belong to
    
    High values of the Modified Hubert Gamma statistic indicate the existence
    of compact clusters.
    
    Parameters
    ----------
    centroids : ndarray of float
        The centroids of the clusters
    clusters : dict
        Cluster id mapped to a list of the cluster's elements
    labels: ndarray of int
        The label of each datapoint, i.e. the cluster id it belongs to
    X : ndarray (2D)
        The data array
    metric : str or callable, optional
        The distance metric to use.  If a string, the distance function can be
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
        'wminkowski', 'yule'.

    Returns
    -------
    Gamma : float
        The Modified Hubert Gamma index
    """
    
    # Compute the matrix Q where Q(i,j) is equal to the distance between the
    # representatives of the clusters where xi and xj belong.
    representatives = centroids.take(labels, axis=0)
    Q = distance.cdist(representatives, representatives, metric=metric)

    # Compute the similarity matrix P of X
    P = distance.pdist(X, metric=metric)
    P = distance.squareform(P)
    
    # Compute the total number of all possible pairs of X
    Gamma = 0
    for i in range(X.shape[0]-1):
        for j in range(i+1, X.shape[0]):
            Gamma += (P[i,j] * Q[i,j])
    M = (X.shape[0] * (X.shape[0] - 1)) / 2

    # Compute and return the statistic
    return (1 / M) * Gamma

def modified_Gamma_index3D(centroids, clusters, labels, X, metric='euclidean',
                           mode=1):
    """
    Return the Modified Hubert Gamma statistic for the 3D structure.
    
    This statistic is calculated using the formula:
    Gamma = (1/M) * sum(sum(P(i,j)*Q(i,j)))
    where:
        M is the number of all possible pairs of X datapoints
        P is the similarity (distance) matrix of X datapoints
        Q is the matrix whose element Q(i,j) holds the distances between the
        centroids of the clusters that xi and xj belong to
    
    High values of the Modified Hubert Gamma statistic indicate the existence
    of compact clusters.
    
    Parameters
    ----------
    centroids : ndarray of float
        The centroids of the clusters
    clusters : dict
        Cluster id mapped to a list of the cluster's elements
    labels: ndarray of int
        The label of each datapoint, i.e. the cluster id it belongs to
    X : ndarray (3D)
        The data array
    metric : str or callable, optional
        The distance metric to use.  If a string, the distance function can be
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
        'wminkowski', 'yule'.
    mode : {0, 1, 2}, default 1
        Determines the axis along which to do the clustering.
        mode=i means that the clustering will be done along axis i.

    Returns
    -------
    Gamma : float
        The Modified Hubert Gamma index
    """
    
    # Compute the matrix Q where Q(i,j) is equal to the distance between the
    # representatives of the clusters where xi and xj belong.
    representatives = centroids.take(labels, axis=0)
    Q = distance.cdist(representatives, representatives, metric=metric)

    # Compute the similarity matrix P of X
    if (mode == 0):
        X_mean = np.nanmean(X, axis=1)
    elif (mode == 1):
        X_mean = np.nanmean(X, axis=0)
    else:
        X_mean= np.swapaxes(np.nanmean(X, axis=0), 0, 1)
    P = distance.pdist(X_mean, metric=metric)
    P = distance.squareform(P)
    
    # Compute the total number of all possible pairs of X
    Gamma = 0
    for i in range(X_mean.shape[0]-1):
        for j in range(i+1, X_mean.shape[0]):
            Gamma += (P[i,j] * Q[i,j])
    M = (X_mean.shape[0] * (X_mean.shape[0] - 1)) / 2

    # Compute and return the statistic
    return (1 / M) * Gamma
