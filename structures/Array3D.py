# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance


class Array3D():
    """
    A 3-dimensional array structure of the form
        n_chromosomes x n_genes x n_samples

    Attributes
    ----------
    X : list of ndarray
        A list containing all genes grouped by chromosome.
    indices : list of lists
        A list containing n_chromosomes lists with the indices of the genes
        in the corresponding chromosomes in X.
    prev_chrom : int
        The index of the previous chromosome examined.
    metric : str or callable, optional
        The distance metric to use.  If a string, the distance function can be
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
        'wminkowski', 'yule'.
    method : {'onebyone', 'centroids'}, default 'centroids'
        The method to be used in order to group together the genes.
        'centroids' means that every time we examine the genes in
        chromosome i, we compute the distances of every gene to the set
        centroids we've already found.
        'onebyone' means that every time we examine the genes in
        chromosome i, we compute the distances of every gene to those in
        chromosome i-1.
    sets : ndarray
        This attribute contains either the last genes of each set (if method is
        'onebyone') or the centroids of each set (if method is 'centroids').
    centroids : dict, int->list of tuples
        This dictionary holds the set id as key and a list containing tuples
        of the form (chromosome, gene_index) for each gene in the set.
    array3D : ndarray
        The final 3D array structure.
    indices3D : ndarray
        The gene labelsof the final 3D array structure.
    array3D_size : int
        The final 3D array's size of the gene axis.

    Methods
    -------
    create(X, metric='euclidean', method='centroids'):
        Create the 3D array from the data in X.
    """

    def __init__(self, X, indices=None):
        """
        Parameters
        ----------
        X : list of ndarray
            A list containing all genes grouped by chromosome.
        indices : list of lists
            A list containing n_chromosomes lists with the indices of the genes
            in the corresponding chromosomes in X.
        """
        if (X is None):
            raise Exception("Error: No data provided!")
        else:
            self.X = X.copy()   # list of ndarray
            self.array3D_size = max([self.X[x].shape[0] for x in range(len(self.X))])
            self.indices = indices.copy()

    def _group_genes(self, chrom):
        """
        Group genes in given chrom to sets.

        Parameters
        ----------
        chrom : int
            The chromosome of which to group the genes.
        """
        distances = np.ma.masked_array(distance.cdist(self.X[chrom],
                                                      self.sets,
                                                      self.metric))

        n_genes_chrom = self.X[chrom].shape[0]
        n_genes_prev = self.sets.shape[0]
        genes_chrom = np.array([-1] * n_genes_chrom)
        genes_prev = np.array([-1] * n_genes_prev)
        n_iterations = min(n_genes_chrom, n_genes_prev)
        for i in range(n_iterations):
            closest = np.unravel_index(distances.argmin(), distances.shape)
            distances[closest[0], :] = np.ma.masked
            distances[:, closest[1]] = np.ma.masked

            # Assign new gene to the appropriate set
            if (self.method == 'centroids'):
                self.centroids[closest[1]].append((chrom, closest[0]))
            self.sets[closest[1]] = self.X[chrom][closest[0]]
            genes_chrom[closest[0]] = closest[1]
            genes_prev[closest[1]] = closest[0]

        if (n_genes_chrom > n_genes_prev):
            # Add the rest of the genes of the current chromosome
            genes_left = np.where(genes_chrom == -1)[0]
            self.sets = np.vstack((self.sets,
                                   self.X[chrom].take(genes_left, axis=0)))
            if (self.method == 'centroids'):
                set_id = n_genes_prev
                for gene_left in genes_left:
                    self.centroids[set_id].append((chrom, gene_left))
                    set_id += 1

        # Update the 3D array
        new_slice = self.sets.copy()
        new_slice[np.where(genes_prev == -1)] = [np.nan] * self.X[chrom].shape[1]
        self.array3D = np.dstack((self.array3D, new_slice))

        # Update the indices of the 3D array
        if (self.indices is not None):
            if (n_genes_chrom > n_genes_prev):
                chrom_ind = np.append(genes_prev, genes_left).tolist()
            else:
                chrom_ind = genes_prev.tolist()
            for i in range(len(chrom_ind)):
                if (chrom_ind[i] != -1):
                    chrom_ind[i] = self.indices[chrom][chrom_ind[i]]
                else:
                    chrom_ind[i] = None
            self.indices3D += [chrom_ind]

    def _reevaluate_centroids(self):
        """
        Calculates the centroids of every slice.
        """
        new_centroids = []

        for set_id in self.centroids.keys():
            set_centroid = []
            for set_gene in self.centroids[set_id]:
                set_centroid.append(self.X[set_gene[0]][set_gene[1]])
            new_centroids.append(np.mean(np.asarray(set_centroid), axis=0))

        self.sets = np.asarray(new_centroids)

    def create(self, metric='euclidean', method='centroids'):
        """
        Create the 3D array with the data in X.

        Parameters
        ----------
        metric : {'euclidean', 'cityblock', 'cosine', 'correlation',
              'hamming', 'jaccard', 'chebyshev', 'canberra',
              'braycurtis', 'yule', 'matching', 'dice',
              'kulsinski', 'rogerstanimoto', 'russellrao',
              'sokalsneath', 'wminkowski', 'fractional'}, default 'euclidean'
              The distance metric to be used
        method : {'centroids', 'onebyone'}, default 'centroids'
            The method to be used in order to group together the genes.
            'centroids' means that every time we examine the genes in
            chromosome i, we compute the distances of every gene to the set
            centroids we've already found.
            'onebyone' means that every time we examine the genes in
            chromosome i, we compute the distances of every gene to those in
            chromosome i-1.

        Returns
        -------
        array3D : ndarray
            The final 3D array structure.
        indices3D : ndarray
            The gene labelsof the final 3D array structure.
        """
        self.metric = metric
        self.method = method

        # Initialize 3D array
        self.sets = np.array(self.X[0])
        self.array3D = self.X[0]
        if (self.indices is not None):
            self.indices3D = [self.indices[0]]
        else:
            self.indices3D = None

        # Add padding to both array3D and indices
        padding = np.array(self.X[0].shape[1] * [np.nan])[np.newaxis, :]
        for i in range(self.array3D_size - self.X[0].shape[0]):
            self.array3D = np.vstack((self.array3D, padding))

        if (method == 'onebyone'):
            for i in range(1, len(self.X)):
                self._group_genes(i)
        else:
            self.centroids = {}
            for i in range(self.array3D_size):
                if (i < self.X[0].shape[0]):
                    self.centroids[i] = [(0, i)]
                else:
                    self.centroids[i] = []
            for i in range(1, len(self.X)):
                self._group_genes(i)
                self._reevaluate_centroids()

        self.array3D = self.array3D.swapaxes(0, 2)
        self.array3D = self.array3D.swapaxes(1, 2)
        # Referece as: [chrom, gene, sample]

        return self.array3D, self.indices3D

if __name__ == '__main__':
    # For debugging purposes

    a = [np.asarray([[5,4,2,1],[6,2,1,4],[8,4,2,4]], dtype=float),
         np.asarray([[8,4,3,2],[6,7,8,4],[1,2,2,4],[5,6,8,3],[4,6,7,3]], dtype=float),
         np.asarray([[9,4,2,7],[8,4,3,7]], dtype=float),
         np.asarray([[6,3,5,1],[6,3,2,5],[8,7,4,3]], dtype=float)]
    a_ind = [[(0, 'a'), (0, 'b'), (0, 'c')],
             [(1, 'd'), (1, 'e'), (1, 'f'), (1, 'g'), (1, 'h')],
             [(2, 'i'), (2, 'j')],
             [(3, 'k'), (3, 'l'), (3, 'm')]]

    arr, ind = Array3D(a, indices=a_ind).create(method='centroids')
