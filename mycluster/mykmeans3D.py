# -*- coding: utf-8 -*-

import numpy as np
import random
from scipy.spatial import distance

import matplotlib.pyplot as plt

import warnings


class KMeans3D():
    """
    The k-means clustering algorithm.

    Attributes
    ----------
    X : ndarray
        The data to be clustered
    n_chroms : int
        The number of chromosomes
    n_genes : int
        The number of genes
    n_samples : int
        The number of samples
    K : int
        The number of clusters to be determined
    order : {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, default 2
        Order of the norm to be used as distance metric.
        See numpy.linalg.norm documentation for more details.
    mode : {0, 1, 2}, default 1
        Determines the axis along which to do the clustering.
        mode=i means that the clustering will be done along axis i.
    mu : ndarray
        The cluster centers
    oldmu : ndarray
        The previous values of cluster centers
    clusters : dict
        Cluster id mapped to a list of the cluster's elements
    method : {'random', 'k-means++'}
        The method of cluster centers initialization
    tol : float
        The algorithm's tolerance
    best_inertia : float
        The algorithm's inertia
    best_clusters : dict
        Same as clusters, but keeps the best clustering we've found so far
    best_labels : ndarray
        Same as labels but keeps the best clustering we've found so far
    best_mu : ndarray
        Same as mu but keeps the best clustering we've found so far
    distances : ndarray
        A 2xm array containg for each data point its closest cluster center
        and the distance from it
    fs : list of float
        The f(k) values for each k
    D2 : ndarray
        Vector of the D^2 weighting for the k-means++ algorithm.


    Methods
    -------
    find_centers(method='random', order=2, mode=1, n_times=10,
                 tol=1e-4, max_iter=300, K=1, verbose=False, init_centers=None)
        Run the k-means3D algorithm with the given parameters.
    get_clusters()
        Return a dictionary cluster_id->list_of_cluster_elements
    get_labels()
        Return an ndarray containing the cluster id for each element
    get_best_k()
        Return a list of the most appropriate values of k
    get_centers()
        Return an ndarray containing the cluster centers
    get_centroids()
        Return an ndarray containing the cluster centroids
    fk()
        Run the f(k) method in order to determine suitable values for k
    plot_fk()
        Plot the values of f(k) against the corresponding values of k

    References
    ----------
    .. [1] Lloyd, Stuart P. "Least Squares Quantization in PCM."
           IEEE Transactions on Information Theory. Vol. 28, 1982, pp. 129–137
    .. [2] Arthur, David, and Sergi Vassilvitskii. "K-means++: The Advantages
           of Careful Seeding." SODA ‘07: Proceedings of the Eighteenth Annual
           ACM-SIAM Symposium on Discrete Algorithms. 2007, pp. 1027–1035
    .. [3] Pham D. T., Dimov S. S., and Nguyen C. D. "Selection of K in
           k-means clustering." DOI: 10.1243/095440605X8298, 2004
    .. [4] Davies D., Bouldin D. "A cluster separation measure", IEEE
           Transactions on Pattern Analysis and Machine Intelligence, Vol. 1,
           pp. 224-227, doi: 10.1109/TPAMI.1979.4766909
    """

    def __init__(self, X):
        """
        Parameters
        ----------
        X : ndarray (3D Array)
            The data to be clustered

        Notes
        -----
        Updates the X, n_chroms, n_genes, n_samples, N, mu, clusters
        and method attributes.
        """
        if (X is None):
            raise Exception("Error: No data provided!")
        else:
            self.X = X

        self.mu = None  # ndarray
        self.clusters = None
        self.method = None

        self.n_chroms = X.shape[0]
        self.n_genes = X.shape[1]
        self.n_samples = X.shape[2]

    def _cluster_points(self):
        """
        Compute the distance of each data point from all cluster centers
        and find the nearest.

        Returns
        -------
        bool
            True if an empty cluster was found, False otherwise.

        Notes
        -----
        Updates the clusters, labels and distances attributes.
        """
        clusters = {}

        X = self.X
        mu = self.mu

        distances = np.empty((1, 2))
        for i in range(X.shape[self.mode]):  # For each slice of X
            if (self.mode == 0):
                distances_x = np.linalg.norm(np.absolute(np.nan_to_num(X[i,:,:].reshape(1, self.n_genes, self.n_samples)-mu)),
                                             ord=self.order, axis=(1,2))
            elif (self.mode == 1):
                distances_x = np.linalg.norm(np.absolute(np.nan_to_num(X[:,i,:].reshape(self.n_chroms, 1, self.n_samples)-mu)),
                                             ord=self.order, axis=(0,2))
            else:
                distances_x = np.linalg.norm(np.absolute(np.nan_to_num(X[:,:,i].reshape(self.n_chroms, self.n_genes, 1)-mu)),
                                             ord=self.order, axis=(0,1))

            distances = np.vstack((distances, [np.nanmin(distances_x),
                                               np.nanargmin(distances_x)]))

        distances = distances[1:, :]
        empty_cluster_found = False
        for clust in range(self.K):
            clusters[clust] = np.where(distances[:, 1] == clust)[0].tolist()
            if (len(clusters[clust]) == 0):
                empty_cluster_found = True

        self.clusters = clusters
        self.labels = np.asarray(distances[:, 1], dtype=int)
        self.distances = distances

        return empty_cluster_found

    def _reevaluate_centers(self):
        """
        Re-evaluates the cluster centers for the Update phase of the algorithm.

        Notes
        -----
        Updates the mu attribute.
        """
        clusters = self.clusters
        if (self.mode == 0):
            newmu = np.empty((1, self.n_genes, self.n_samples))
        elif (self.mode == 1):
            newmu = np.empty((self.n_chroms, 1, self.n_samples))
        else:
            newmu = np.empty((self.n_chroms, self.n_genes, 1))

        keys = sorted(self.clusters.keys())
        for k in keys:
            with warnings.catch_warnings():  # We expect mean of NaNs here
                warnings.simplefilter("ignore", category=RuntimeWarning)
                newmu_i = np.nanmean(self.X.take(clusters[k], axis=self.mode),
                                     axis=self.mode)  # Could contain NaNs

            if (self.mode == 0):
                newmu_i = newmu_i.reshape((1, self.n_genes, self.n_samples))
                newmu = np.vstack((newmu, newmu_i))
            elif (self.mode == 1):
                newmu_i = newmu_i.reshape((self.n_chroms, 1, self.n_samples))
                newmu = np.hstack((newmu, newmu_i))
            else:
                newmu_i = newmu_i.reshape((self.n_chroms, self.n_genes, 1))
                newmu = np.dstack((newmu, newmu_i))

        if (self.mode == 0):
            self.mu = newmu[1:, :, :]
        elif (self.mode == 1):
            self.mu = newmu[:, 1:, :]
        else:
            self.mu = newmu[:, :, 1:]

    def _has_converged(self):
        """
        Returns True if the algorithm has converged, i.e. the difference
        between new cluster centers and previous ones is less than the
        given tolerance.

        Returns
        -------
        bool
            True if the algorithm has converged, False otherwise.
        
        """
        mu = np.nan_to_num(self.mu)
        oldmu = np.nan_to_num(self.oldmu)
        d = np.ravel(oldmu - mu)
        return np.dot(d, d) <= self.tol

    def find_centers(self, method='random', order=2, mode=1, n_times=10,
                     tol=1e-4, max_iter=300, K=1, verbose=False,
                     init_centers=None):
        """
        Run the k-means3D algorithm with the given parameters.

        Parameters
        ----------
        method : {'random', 'k-means++'}, default 'random'
            The method to be used for cluster centers initialization
        order : {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, default 2
            Order of the norm to be used as distance metric.
            See numpy.linalg.norm documentation for more details.
        mode : {0, 1, 2}, default 1
            Determines the axis along which to do the clustering.
            mode=i means that the clustering will be done along axis i.
        n_times : int, default 10
            Number of times for the k-means algorithm to be run
        tol : float, default 1e-4
            The tolerance. If the difference between the previous cluster
            centers and the new ones is less than the tolerance, then the
            algorithm converges
        max_iter : int, default 300
            The maximum number of iterations for the algorithm
        K : int, default 1
            The number of clusters
        verbose : bool, default False
            Verbose mode for debugging
        init_centers : ndarray
            The user can give an initialization of the cluster centers to be
            used by the algorithm.

        Notes
        -----
        Updates the K, method, metric, tol, best_inertia, best_clusters,
        best_labels, best_mu, distances, labels, clusters and oldmu attributes.
        """
        self.K = K
        self.method = method
        self.order = order
        self.mode = mode
        self.tol = tol

        # Check the parameters given
        if (len(self.X) == 0):
            raise ValueError("Please provide valid data to the algorithm.")
        if (floor(self.K) != self.K) or (self.K <= 0):
            raise ValueError("'k' must be an integer > 0, but its value"
                             " is {}".format(self.K))
        if (self.method not in ['random', 'k-means++']):
            raise ValueError("'method' must be either 'random' or 'k-means++',"
                             " but its value is {}".format(self.delta))
        if (floor(max_iter) != max_iter) or (max_iter < 0):
            raise ValueError("'max_iter' must be an integer > 0, but its value"
                             " is {}".format(self.K))

        self.best_inertia = None
        self.best_clusters = None
        self.best_labels = None
        self.best_mu = None

        self.distances = np.asarray([[None], [None]])
        self.labels = None
        self.clusters = None

        # Run k-means algorithm n_times
        for i in range(n_times):
            if (verbose):
                print(i)

            self.oldmu = self.X.take(np.random.randint(self.X.shape[self.mode],
                                                       size=self.K),
                                     axis=self.mode)  # Could contain NaNs

            # Initialize centers
            if (init_centers is None):
                self._init_centers()
            else:
                self.mu = init_centers

            # E-M steps
            for j in range(max_iter):
                if (not self._has_converged()):
                    self.oldmu = self.mu
                    # Assign all points in X to clusters
                    empty_cluster_found = self._cluster_points()
                    if (empty_cluster_found):
                        print("   Empty cluster found! Exiting...")
                        break
                    # Reevaluate centers
                    self._reevaluate_centers()
                else:
                    if (verbose):
                        print("   Converged in iteration {}".format(j))
                    break

            # Calculate inertia (Sum of distances of samples to their closest
            # cluster center)
            if (not empty_cluster_found):
                inertia = np.sum(self.distances[:, 0])
                if (self.best_inertia is None) or (inertia < self.best_inertia):
                    self.best_inertia = inertia
                    self.best_labels = self.labels
                    self.best_clusters = self.clusters
                    self.best_mu = self.mu

    def get_clusters(self):
        """
        Return the clusters formed.

        Returns
        -------
            best_clusters : dict
                Dictionary of the form: cluster_id->list_of_points_of_cluster
        """
        return self.best_clusters

    def get_labels(self):
        """
        Return the labels found.

        Returns
        -------
        best_labels : ndarray
            The id of the cluster that each data point belongs to
        """
        return np.array(self.best_labels)

    def get_centers(self):
        """
        Return the values of the cluster centers.

        Returns
        -------
        best_mu : ndarray
            The values of the cluster centers found
        """
        return self.best_mu

    def get_centroids(self):
        """
        Return the cluster centroids.

        Returns
        -------
        centroids : ndarray
            The cluster centroids
        """
        if (self.best_mu is None):
            return None
        with warnings.catch_warnings():  # We expect mean of NaNs here
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if (self.mode == 0):
                return np.nanmean(self.best_mu, axis=1)
            elif (self.mode == 1):
                return np.nanmean(self.best_mu, axis=0)
            else:
                return np.rollaxis(np.nanmean(self.best_mu, axis=0), 1)

    def get_best_k(self):
        """
        Return the most suitable values for k.

        Returns
        -------
        best_k : list of int
            The values of k in priority order (those with the smallest f(k) are
            listed first)
        """
        return [i+1 for i in np.argsort(self.fs)]

    def _dist_from_centers(self):
        """
        Calculate the distance of each data point from each cluster center.

        Notes
        -----
        Updates the D2 attribute.
        """
        X = self.X
        cent = self.mu
        D2 = np.array([])

        for i in range(X.shape[self.mode]):  # For each slice of X
            if (self.mode == 0):
                D2_x = np.power(np.linalg.norm(np.absolute(np.nan_to_num(X[i,:,:].reshape(1,self.n_genes,self.n_samples)-cent)),
                                               ord=self.order, axis=(1,2)), 2)
            elif (self.mode == 1):
                D2_x = np.power(np.linalg.norm(np.absolute(np.nan_to_num(X[:,i,:].reshape(self.n_chroms,1,self.n_samples)-cent)),
                                               ord=self.order, axis=(0,2)), 2)
            else:
                D2_x = np.power(np.linalg.norm(np.absolute(np.nan_to_num(X[:,:,i].reshape(self.n_chroms, self.n_genes, 1)-cent)),
                                               ord=self.order, axis=(0,1)), 2)
            D2 = np.hstack((D2, np.nanmin(D2_x)))

        self.D2 = np.nan_to_num(D2)

    def _choose_next_center(self):
        """
        Compute and update the values of the cluster centers.

        Returns
        -------
        ndarray of float
            An array containing the new cluster centers
        """
        probs = self.D2/self.D2.sum()
        cumprobs = probs.cumsum()
        while True:
            r = random.random()
            ind = np.where(cumprobs >= r)[0][0]
            if (probs[ind] != 0):
                break

        if (self.mode == 0):
            return self.X[ind,:,:].reshape((1, self.n_genes, self.n_samples))
        elif (self.mode == 1):
            return self.X[:,ind,:].reshape((self.n_chroms, 1, self.n_samples))
        else:
            return self.X[:,:,ind].reshape((self.n_chroms, self.n_genes, 1))

    def _init_centers(self):
        """
        Initialize the cluster centers according to given method.

        Notes
        -----
        Updates the mu attribute.
        Caution: Mu might contain NaNs!
        """
        if (self.method == 'k-means++'):   # k-means++ initialization
            self.mu = self.X.take(np.random.randint(self.X.shape[self.mode]),
                                  axis=self.mode)

            if (self.mode == 0):
                self.mu = self.mu.reshape((1, self.mu.shape[0], self.mu.shape[1]))
            elif (self.mode == 1):
                self.mu = self.mu.reshape((self.mu.shape[0], 1, self.mu.shape[1]))
            else:
                self.mu = self.mu.reshape((self.mu.shape[0], self.mu.shape[1], 1))
            i = 1
            while (i < self.K):
                self._dist_from_centers()
                if (self.mode == 0):
                    self.mu = np.vstack((self.mu, self._choose_next_center()))
                elif (self.mode == 1):
                    self.mu = np.hstack((self.mu, self._choose_next_center()))
                else:
                    self.mu = np.dstack((self.mu, self._choose_next_center()))
                i += 1
        else:  # random initialization
            self.mu = self.X.take(np.random.randint(self.X.shape[self.mode],
                                                    size=self.K),
                                  axis=self.mode)

    def _memoize(func):
        """
        Memoization wrapper function, for  optimization purposes.
        """
        memo = {}

        def helper(*args):
            if args not in memo:
                memo[args] = func(*args)
            return memo[args]

        return helper

    @_memoize
    def _ak(self, k, Nd):
        """
        Compute the weight factor for f(k).

        Parameters
        ----------
        k : int
            The current number of clusters
        Nd : int
            The dimensions of the data

        Returns
        -------
        a_k : float
            The weight factor
        """
        if (k == 2):
            a_k = 1.0 - (3.0 / (4.0 * Nd))
        else:
            a_k = self._ak(k-1, Nd) + (1.0 - self._ak(k-1, Nd)) / 6.0
        return a_k

    def fk(self, maxk, method='random', order=2, mode=1, n_times=10,
           max_iter=300, tol=1e-4, init_fs=None, verbose=False):
        """
        Compute the evaluation function f(k) for different values of k, in
        order to determine the most suitable number of clusters.

        Parameters
        ----------
        maxk : int
            The maximum value of `k` to be considered
        method : {'random', 'k-means++'}, default 'random'
            The method to be used for cluster centers initialization
        order : {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, default 2
            Order of the norm to be used as distance metric.
            See numpy.linalg.norm documentation for more details.
        mode : {0, 1, 2}, default 1
            Determines the axis along which to do the clustering.
            mode=i means that the clustering will be done along axis i.
        n_times : int, default 10
            Number of times for the k-means algorithm to be run
        max_iter : int, default 300
            The maximum number of iterations for the algorithm
        tol : float, default 1e-4
            The tolerance. If the difference between the previous cluster
            centers and the new ones is less than the tolerance, then the
            algorithm converges
        init_fs : ndarray of float
            The user can give a list of f(k) values for k, 1 <= k <= i, and
            the function will proceed to find the next f(k) values for
            i < k <= maxk
        verbose : bool, default False
            Verbose mode for debugging

        Notes
        -----
        Updates the metric, method, K and fs attributes.
        """
        fs = np.zeros(maxk)

        X = self.X
        if (mode == 0) or (mode == 1):
            Nd = self.n_genes
        else:
            Nd = self.n_samples

        if (init_fs is None):
            start_k = 1
            Skm1 = 0
        else:
            start_k = len(init_fs)
            fs[:start_k] = init_fs

            # Compute Skm1
            self.K = start_k-1
            self.find_centers(K=start_k-1, order=order, max_iter=max_iter,
                              mode=mode, n_times=n_times, tol=tol,
                              method=method)
            centroids = self.get_centroids()
            clusters = self.best_clusters

            Skm1 = 0
            for i in sorted(clusters.keys()):
                i_data = X.take(clusters[i], axis=self.mode)

                if (self.mode == 0) or (self.mode == 1):
                    # Convert i_data to 2-D array by flattening the chromosomes
                    # dimension
                    i_data = np.nan_to_num(np.reshape(i_data, (i_data.shape[0]*i_data.shape[1],
                                                               i_data.shape[2])))
                    # Remove the NaN rows
                    i_data = i_data.take(np.unique(np.nonzero(i_data)[0]), axis=0)
                else:
                    # Take the mean of each sample for all chromosomes
                    i_data = np.nan_to_num(np.nanmean(i_data, axis=0))

                    # Swap axes 1 and 2
                    i_data = np.swapaxes(i_data, 0, 1)

                # Compute Skm1
                Skm1 = Skm1 + np.sum(np.square(distance.cdist(np.nan_to_num(centroids[i, np.newaxis]),
                                                              i_data)))

        # Compute fs and Sk for the next k values
        for k in range(start_k, maxk+1):
            if (verbose):
                print("k = {}".format(k))

            self.K = k
            self.find_centers(K=k, order=order, max_iter=max_iter, mode=mode,
                              n_times=n_times, tol=tol, method=method)
            centroids = self.get_centroids()
            clusters = self.best_clusters

            Sk = 0
            for i in sorted(clusters.keys()):

                i_data = X.take(clusters[i], axis=self.mode)

                if (self.mode == 0) or (self.mode == 1):
                    # Convert i_data to 2-D array by flattening the chromosomes
                    # dimension
                    i_data = np.nan_to_num(np.reshape(i_data, (i_data.shape[0]*i_data.shape[1],
                                                               i_data.shape[2])))
                    # Remove the NaN rows
                    i_data = i_data.take(np.unique(np.nonzero(i_data)[0]), axis=0)
                else:
                    # Take the mean of each sample for all chromosomes
                    i_data = np.nan_to_num(np.nanmean(i_data, axis=0))

                    # Swap axes 1 and 2
                    i_data = np.swapaxes(i_data, 0, 1)

                # Compute Sk
                Sk = Sk + np.sum(np.square(distance.cdist(np.nan_to_num(centroids[i, np.newaxis]),
                                                          i_data)))

            if (k == 1):
                fs[0] = 1
            elif (Skm1 == 0):
                fs[k-1] = 1
            else:
                fs[k-1] = Sk / (self._ak(k, Nd)*Skm1)
                
            Skm1 = Sk

        self.fs = fs

    def plot_fk(self, maxk):
        """
        Plot the computed f(k) values against the corresponding k values.

        Parameters
        ----------
        maxk : int
            The maximum value of k considered
        """
        fig = plt.figure()
        plt.plot(range(1, maxk+1), self.fs, 'ro-', alpha=0.6)
        print(self.fs)
        plt.xlabel('Number of clusters K', fontsize=16)
        plt.ylabel('f(K)', fontsize=16)
        foundfK = np.where(self.fs == min(self.fs))[0][0] + 1
        tit2 = 'f(K) finds %s clusters' % (foundfK)
        plt.title(tit2, fontsize=16)
        fig.show()

if __name__ == '__main__':
    # For debugging purposes

#    from mpl_toolkits.mplot3d import Axes3D
    from sympy import ceiling
    from Array3D import Array3D

    np.random.seed(0)
    data = []
    for i in range(20):
        data.append((10 - (-10)) * np.random.random_sample((100, 10)) + (-10))

    arr = Array3D(data).create()

    for i in range(20):
        for j in range(np.random.randint(0, high=15)):
            arr[i, np.random.randint(0, high=100), :] = [np.nan] * 10
            
    for i in range(100):
        if (np.isnan(arr[:, i, 0]).all()):
            arr[np.random.randind(0, high=3), i, :] = 20 * np.random.random_sample((100, 10)) + (-10)

    # Run 3D K-Means
    kmeans_model = KMeans3D(arr)

    mode = 1

    maxk = 90
#    kmeans_model.fk(maxk, method='k-means++', order='fro', mode=mode,
#                    verbose=True)
#    best_k = kmeans_model.get_best_k()
#
#    print(best_k)
#
#    kmeans_model.plot_fk(maxk)

    best_k = [1, 1, 1, 1, maxk]
    # Run k-means for the first 5 best values of k
    for k in best_k[:5]:
        if (k == 1):
            continue
        kmeans_model.find_centers(method='k-means++', order='fro', mode=mode,
                                  K=k, tol=1e-5, verbose=True)
        labels = kmeans_model.get_labels()
        centroids = kmeans_model.get_centroids()

        # Plot clusters separately
        no_windows = (k // 10)
        if (k % 10 > 0):
            no_windows += 1

        clust = 0
        for win in range(1, no_windows+1):
            fig = plt.figure("k=" + str(k) + ", part " + str(win))
            while (clust < k) and (clust < win*10):
    #            plt.xlabel('Samples')
    #            plt.ylabel('Gene expression')
                if ((k // 10) - win < 0):
                    plt.subplot(ceiling((k % 10) / 2.0), 2, (clust % 10)+1)
                else:
                    plt.subplot(5, 2, (clust % 10)+1)

                cluster_genes = np.where(labels == clust)[0]
                plt.title('Cluster ' + str(clust) + ' (' + str(len(cluster_genes)) + ')')
                for gene_group in cluster_genes:  # for each gene in the cluster
                    if (mode == 0):
                        for gene in arr[gene_group, :, :]:
                            plt.plot(range(arr.shape[2]), gene)
                    elif (mode == 1):
                        for gene in arr[:, gene_group, :]:
                            plt.plot(range(arr.shape[2]), gene)
                    else:
                        for sample in arr[:, :, gene_group]:
                            plt.plot(range(arr.shape[1]), sample)
                if (mode == 0):
                    plt.plot(range(arr.shape[2]), centroids[clust, :], color='k',
                             linewidth=2)
                elif (mode == 1):
                    plt.plot(range(arr.shape[2]), centroids[clust, :], color='k',
                             linewidth=2)
                else:
                    plt.plot(range(arr.shape[1]), centroids[clust, :], color='k',
                             linewidth=2)

                clust += 1

            fig.show()
