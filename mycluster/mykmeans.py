# -*- coding: utf-8 -*-

import numpy as np
import random
from scipy.spatial import distance
from scipy import floor

import time
import matplotlib.pyplot as plt


class KMeans():
    """
    The k-means clustering algorithm.

    Attributes
    ----------
    X : ndarray
        The data to be clustered
    K : int
        The number of clusters to be determined
    N : int
        The dimension of the data
    mu : ndarray
        The cluster centers
    oldmu : ndarray
        The previous values of cluster centers
    clusters : dict
        Cluster id mapped to a list of the cluster's elements
    method : {'random', 'k-means++'}
        The method of cluster centers initialization
    metric : str or callable, optional
        The distance metric to use.  If a string, the distance function can be
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
        'wminkowski', 'yule', 'fractional'.
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
    f : float, optional
        The order of the fractional norm
    fs : list of float
        The f(k) values for each k
    D2 : ndarray
        Vector of the D^2 weighting for the k-means++ algorithm.
        It is a (1 x n_genes) vector containing the shortest (squared )distance
        of each gene from a centroid.


    Methods
    -------
    find_centers(method='random', metric='euclidean', f=None,
                 n_times=10, tol=1e-4, max_iter=300, K=1, verbose=False)
        Run the k-means algorithm with the given parameters.
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
        X : ndarray
            The data to be clustered

        Notes
        -----
        Updates the X, N, mu, clusters and method attributes.
        """
        if (X is None):
            raise Exception("Error: No data provided!")
        else:
            self.X = X   # ndarray
            self.N = len(X)  # Number of genes

        self.mu = None  # ndarray
        self.clusters = None
        self.method = None

    def _cluster_points(self):
        """
        Compute the distance of each data point from all cluster centers
        and find the nearest.

        Notes
        -----
        Updates the clusters, labels and distances attributes.
        """
        mu = self.mu
        clusters = {}
        X = self.X

        if (self.metric == 'fractional'):
            distances = np.asarray(distance.cdist(X,
                                                  mu,
                                                  lambda x1, x2: np.linalg.norm(x1-x2, ord=self.f)))
        else:
            distances = np.asarray(distance.cdist(X, mu, self.metric))

        distances = np.asarray((np.amin(distances, axis=1),
                                np.argmin(distances, axis=1)))

        for clust in range(self.K):
            clusters[clust] = np.where(distances[1] == clust)[0].tolist()

        self.clusters = clusters
        self.labels = np.asarray(distances[1], dtype=int)
        self.distances = distances

    def _reevaluate_centers(self):
        """
        Re-evaluates the cluster centers for the Update phase of the algorithm.

        Notes
        -----
        Updates the mu attribute.
        """
        clusters = self.clusters
        newmu = []
        keys = sorted(self.clusters.keys())
        for k in keys:
            newmu.append(np.mean(np.take(self.X, clusters[k], axis=0), axis=0))
        self.mu = np.asarray(newmu)

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
        mu = self.mu
        oldmu = self.oldmu
        d = np.ravel(oldmu - mu)
        return np.dot(d, d) <= self.tol

    def find_centers(self, method='random', metric='euclidean', f=0.5,
                     n_times=10, tol=1e-4, max_iter=300, K=1, verbose=False,
                     init_centers=None):
        """
        Runs the k-means algorithm with the given parameters.

        Parameters
        ----------
        method : {'random', 'k-means++'}, default 'random'
            The method to be used for cluster centers initialization
        metric : str or callable, optional
            The distance metric to use.  If a string, the distance function can
            be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
            'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard',
            'kulsinski', 'mahalanobis', 'matching', 'minkowski',
            'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
            'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule', 'fractional'.
        f : float, default 0.5
            The order of the norm in case of 'fractional' distance metric
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
            used by the algorithm. Ndarray should be of the form (n_clusters,
            n_features)

        Notes
        -----
        Updates the K, method, metric, tol, best_inertia, best_clusters,
        best_labels, best_mu, distances, labels, clusters and oldmu attributes.
        """
        self.K = K
        self.method = method
        self.metric = metric
        X = self.X.tolist()
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

        if (metric == 'fractional'):
            self.f = f

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

            self.oldmu = np.asarray(random.sample(X, K))

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
                    self._cluster_points()
                    # Reevaluate centers
                    self._reevaluate_centers()
                else:
                    if (verbose):
                        print("   Converged in iteration {}".format(j))
                    break

            # Calculate inertia (Sum of distances of samples to their closest
            # cluster center)
            inertia = np.sum(self.distances[0])
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

    def get_best_k(self):
        """
        Return the most suitable values for k.

        Returns
        -------
        best_k : list of int
            The values of k in priority order (those with the smallest f(k) are
            listed first)
        """
        return [i+1 for i in np.argsort(self.fs) if self.fs[i] < 0.85]

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
        return np.vstack(self.X.take(self.best_clusters[x], axis=0).mean(axis=0)
                         for x in self.best_clusters.keys())

    def _dist_from_centers(self):
        """
        Calculate the distance of each data point from each cluster center.

        Notes
        -----
        Updates the D2 attribute.
        """
        cent = self.mu
        X = self.X

        if (self.metric == 'fractional'):
            D2 = np.amin(np.power(distance.cdist(X,
                                                 cent,
                                                 lambda x1, x2: np.linalg.norm(x1-x2, ord=self.f)),
                                  2),
                         axis=1).flatten(order='A')
        else:
            D2 = np.amin(np.power(distance.cdist(X, cent, self.metric), 2),
                         axis=1).flatten(order='A')

        self.D2 = D2

    def _choose_next_center(self):
        """
        Compute and update the values of the cluster centers.

        Returns
        -------
        list of float
            A list containing the new cluster centers
        """
        probs = self.D2/self.D2.sum()
        cumprobs = probs.cumsum()
        r = random.random()
        ind = np.where(cumprobs >= r)[0][0]
        return self.X[ind].tolist()

    def _init_centers(self):
        """
        Initialize the cluster centers according to given method.

        Notes
        -----
        Updates the mu attribute.
        """
        if (self.method == 'k-means++'):   # k-means++ initialization
            self.mu = np.asarray(random.sample(self.X.tolist(), 1))
            while (len(self.mu) < self.K):
                self._dist_from_centers()
                self.mu = np.vstack((self.mu, self._choose_next_center()))
        else:  # random initialization
            self.mu = np.asarray(random.sample(self.X.tolist(), self.K))

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

    def fk(self, maxk, metric='euclidean', method='random', f=0.5, n_times=10,
           max_iter=300, tol=1e-4, verbose=False):
        """
        Compute the evaluation function f(k) for different values of k, in
        order to determine the most suitable number of clusters.

        Parameters
        ----------
        maxk : int
            The maximum value of `k` to be considered
        metric : str or callable, optional
            The distance metric to use.  If a string, the distance function can
            be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
            'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard',
            'kulsinski', 'mahalanobis', 'matching', 'minkowski',
            'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
            'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule', 'fractional'.
        method : {'random', 'k-means++'}, default 'random'
            The method to be used for cluster centers initialization
        f : float, default 0.5
            The order of the norm in case of 'fractional' distance metric
        n_times : int, default 10
            Number of times for the k-means algorithm to be run
        max_iter : int, default 300
            The maximum number of iterations for the algorithm
        tol : float, default 1e-4
            The tolerance. If the difference between the previous cluster
            centers and the new ones is less than the tolerance, then the
            algorithm converges
        verbose : bool, default False
            Verbose mode for debugging

        Notes
        -----
        Updates the metric, method, K and fs attributes.
        """

        X = self.X
        Nd = self.N

        fs = np.zeros(maxk)

        self.K = 1
        Skm1 = 0

        for k in range(1, maxk+1):
            if (verbose):
                print("k = {}".format(k))

            self.K = k

            self.find_centers(K=k, max_iter=max_iter, n_times=n_times, tol=tol,
                              method=method)
            best_clusters = self.best_clusters
            best_mu = self.best_mu

            Sk = 0
            for clust in best_clusters.keys():
                if (self.metric == 'fractional'):
                    Sk = Sk + \
                         np.sum(np.square(distance.cdist(X.take(best_clusters[clust], axis=0),
                                                         best_mu[clust, np.newaxis],
                                                         lambda x1, x2: np.linalg.norm(x1-x2, ord=f))))
                else:
                    Sk = Sk + \
                         np.sum(np.square(distance.cdist(X.take(best_clusters[clust], axis=0),
                                                         best_mu[clust, np.newaxis],
                                                         metric)))
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
        return self.fs


if __name__ == '__main__':
    # For debugging purposes

    np.random.seed(0)
    m = np.asarray([[0, 0], [4, 0], [0, 4], [5, 4]])
    S0 = np.identity(2)
    S1 = np.asarray([[1, 0.2], [0.2, 1.5]])
    S2 = np.asarray([[1, 0.4], [0.4, 1.1]])
    S3 = np.asarray([[0.3, 0.2], [0.2, 0.5]])

    data = np.random.multivariate_normal(m[0, :], S0, 100)
    data = np.vstack((data, np.random.multivariate_normal(m[1, :], S1, 100)))
    data = np.vstack((data, np.random.multivariate_normal(m[2, :], S2, 100)))
    data = np.vstack((data, np.random.multivariate_normal(m[3, :], S3, 100)))

    fig = plt.figure()
    plt.plot(data[:, 0], data[:, 1], 'bo')
    fig.show()

    kmeans_model = KMeans(data)

    start_time = time.time()

    maxk = 10
    kmeans_model.fk(maxk, method='k-means++', metric='euclidean')
    best_k = kmeans_model.get_best_k()

    print("--- %s seconds ---" % (time.time() - start_time))

    kmeans_model.plot_fk(maxk)

    x = data[:, 0]
    y = data[:, 1]
    colors = ['b', 'r', 'g', 'y', 'c', 'm', 'k', '#a700ee', '#c08a2e', '#b3b3b3']
    for k in best_k:
        kmeans_model.find_centers(method='k-means++', metric='euclidean', K=k)
        labels = kmeans_model.get_labels()

        fig = plt.figure()
        for i in range(k):
            plt.plot(x.take(np.where(labels == i)),
                     y.take(np.where(labels == i)),
                     color=colors[i],
                     marker='x')
        plt.title('k = ' + str(k))
        fig.show()

        # Find the DB index
        print("DB index of k={0} is: {1}".format(k, kmeans_model.DB_index()))
