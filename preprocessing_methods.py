# -*- coding: utf-8 -*-

import numpy as np
from sklearn import neighbors
import pandas as pd


def quantile_normalization(data, method='mean'):
    """
    Quantile normalization of data.
    """
    data_norm = data.copy()
    I = np.argsort(data_norm.ix[:, 1:], axis=0)
    if (method == 'median'):
        data_medians = np.float64(np.nanmedian(data_norm.ix[:, 1:].values[I, np.arange(data_norm.shape[1]-1)],
                                               axis=1)[:, np.newaxis])
        data_norm.ix[:, 1:].values[I, np.arange(data_norm.shape[1]-1)] = data_medians
    else:
        data_means = np.float64(np.nanmean(data_norm.ix[:, 1:].values[I, np.arange(data_norm.shape[1]-1)],
                                           axis=1)[:, np.newaxis])
        data_norm.ix[:, 1:].values[I, np.arange(data_norm.shape[1]-1)] = data_means

    return data_norm


def kNN_imputation(a, k=1, imp_method='mean', metric='euclidean',
                   **metric_params):
    """
    Performs value imputation using the k Nearest Neighbors algorithm.

    For all missing (i.e. NaN) values in data, we find the k Nearest
    Neighbors and then replace the NaN value with a weighted average of the
    found neighbors.

    data is of the form: genes x samples
    Suppose gene i contains a NaN value in sample j. This function chooses k
    genes with non-missing values in sample j, nearest to gene i (i.e. those
    genes that have the closest expression profiles to gene i in the remaining
    samples). Also, only genes with complete columns can be neighbors. Then, it
    uses the average value of those k neighbors in sample j to fill in the
    NaN value.

    Parameters
    ----------
    a : pandas.DataFrame
        The data in which we will perform the iputation.
    k : int, optional
        The number of nearest neighbors to take into account.
    imp_method : {'mean', 'median'}, optional
                The method of imputation.
    metric : {'cityblock', 'cosine', 'euclidean', 'l1', 'l2',
              'manhattan', 'braycurtis', 'canberra', 'chebyshev',
              'correlation', 'dice', 'hamming', 'jaccard',
              'kulsinski', 'mahalanobis', 'matching', 'minkowski',
              'rogerstanimoto', 'russellrao', 'seuclidean',
              'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'}, optional
              The distance metric to be used in the k Nearest Neighbors
              algorithm.
    metric_params : dict, optional
                    Additional keyword arguments for the metric function.

    Returns
    -------
    pandas.DataFrame
        The given DataFrame with the NaN values imputed.

    References
    ----------
    .. [1] P. Jonsson, C. Wohlin, 2004. An Evaluation of kNearest Neighbour
           Imputation Using Likert Data, Proceedings of the 10th International
           Symposium on Software Metrics, Chicago, IL, (USA), pp. 108 – 118

    .. [2] O. Troyanskaya, M. Cantor, G. Sherlock, P. Brown, T. Hastie,
           R. Tibshirani, D. Botstein, and R. B. Altman, (2001). Missing value
           estimation methods for DNA microarrays, Bioinformatics,
           17 (6): 520-525 doi:10.1093/bioinformatics/17.6.520
    """

    # Check if given parameters are correct
    if (imp_method not in ['mean', 'median']):
        print("Error: Invalid method")
        return
    if (metric not in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
                       'manhattan', 'braycurtis', 'canberra', 'chebyshev',
                       'correlation', 'dice', 'hamming', 'jaccard',
                       'kulsinski', 'mahalanobis', 'matching', 'minkowski',
                       'rogerstanimoto', 'russellrao', 'seuclidean',
                       'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']):
        print("Error: Invalid metric")
        return

    # Get a copy of the original array
    a_imputed = a.values.copy()

    # Find nans in original array
    nanVals = np.isnan(a_imputed)

    # Find rows that do not contain any nans
    noNans = np.logical_not(np.any(nanVals, axis=1))

    # Keep rows from the original array that do not contain any nans
    dataNoNans = a_imputed[noNans, :]

    # If there are no such rows, print error message
    if (dataNoNans.size == 0):
        print("Error: There are no rows without NaN values.")
        return

    # Find indices of nan values in original matrix
    (nan_rows, nan_cols) = np.nonzero(nanVals)

    rowWarn = np.zeros((a_imputed.shape[0], 1))

    nan_rows, slices = np.unique(nan_rows, return_index=True)
    nan_cols = np.split(nan_cols, slices[1:])

    knn = neighbors.NearestNeighbors(n_neighbors=k+1,
                                     metric=metric,
                                     metric_params=metric_params)

    # For each nan row
    for nan_row in range(nan_rows.size):
        # Check if the row contains only nans
        if (np.all(np.isnan(a_imputed[nan_rows[nan_row], :]))):
            if (rowWarn[nan_rows[nan_row]] == 0):
                print("Warning: row {0} contains only NaN values.".format(nan_rows[nan_row]))
                rowWarn[nan_rows[nan_row]] = 1
            continue

        # Find all columns that do not correspond to any nan values of the
        # nan row
        complete_cols = [x for x in range(a_imputed.shape[1]) if x not in nan_cols[nan_row]]

        knn.fit(np.vstack((a_imputed[nan_rows[nan_row], complete_cols],
                           dataNoNans[:, complete_cols])))
        neighs = knn.kneighbors(a_imputed[nan_rows[nan_row], complete_cols].reshape(1, -1),
                                n_neighbors=k+1,
                                return_distance=False)

        # We ignore the first neighbor - it's the reference vector itself
        neighs = neighs[0][1:]
        # Impute values
        for nan_col in nan_cols[nan_row]:
            if (imp_method == 'mean'):
                a_imputed[nan_rows[nan_row], nan_col] = np.mean(a_imputed[neighs-1, nan_col])
            else:
                a_imputed[nan_rows[nan_row], nan_col] = np.median(a_imputed[neighs-1, nan_col])

    a[:] = a_imputed
    return a

