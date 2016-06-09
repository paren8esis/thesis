# -*- coding: utf-8 -*-

import numpy as np

import warnings


class EmptyTriclusterException(Exception):
    pass


class DeltaTrimax():
    """
    The delta-TRIMAX clustering algorithm.

    Attributes
    ----------
    D : ndarray
        The data to be clustered
    delta : float
        The delta parameter of the algorithm. Must be > 0.0
    l : float
        The lambda parameter of the algorithm. Must be >= 1.0
    chrom_cutoff : int
        The deletion threshold for the chromosome axis
    gene_cutoff : int
        The deletion threshold for the gene axis
    sample_cutoff : int
        The deletion threshold for the sample axis
    tol : float
        The algorithm's tolerance
    mask_mode : {'random', 'nan'}
        The masking method for the clustered values. If 'random', the values
        are replaced by random floats. If 'nan', they are replaced by nan
        values.
    n_chroms : int
        The number of chromosome pairs
    n_genes : int
        The number of genes
    n_samples : int
        The number of samples
    result_chroms : list of ndarray
        A list of length #triclusters, containg a boolean ndarray for each
        tricluster. The boolean array is of length #chromosomes and contains
        True if the respective chromosome is contained in the tricluster,
        False otherwise.
    result_genes : list of ndarray
        A list of length #triclusters, containg a boolean ndarray for each
        tricluster. The boolean array is of length #genes and contains
        True if the respective gene is contained in the tricluster,
        False otherwise.
    result_samples : list of ndarray
        A list of length #triclusters, containg a boolean ndarray for each
        tricluster. The boolean array is of length #samples and contains
        True if the respective sample is contained in the tricluster,
        False otherwise.
    MSR : float
        The Mean Squared Residue of each cell.
    MSR_chrom : float
        The Mean Squared Residue of each chromosome.
    MSR_gene : float
        The Mean Squared Residue of each gene.
    MSR_sample : float
        The Mean Squared Residue of each sample.

    Methods
    -------
    fit(self, delta=2.5, l=1.005, chrom_cutoff=50, gene_cutoff=50,
            sample_cutoff=50, tol=1e-5, mask_mode='nan', verbose=False)
        Run the delta-TRIMAX algorithm for the given parameters.
    get_triclusters()
        Return the triclusters found by the algorithm.

    References
    ----------
    .. [1] A. Bhar, M. Haubrock, A. Mukhopadhyay, U. Maulik, S. Bandyopadhyay,
           and E. Wingender, ‘Coexpression and coregulation analysis of
           time-series gene expression data in estrogen-induced breast cancer
           cell’, Algorithms Mol. Biol., τ. 8, τχ. 1, σ 9, 2013.
    """

    def __init__(self, D):
        """
        Parameters
        ----------
        D : ndarray
            The data to be clustered
        """
        self.D = D.copy()

    def _check_parameters(self):
        """
        Checks the parameters given by the user. If the values are not valid,
        a ValueError is raised.
        """
        if (self.delta < 0):
            raise ValueError("'delta' must be > 0.0, but its value"
                             " is {}".format(self.delta))
        if (self.l < 1):
            raise ValueError("'lambda' must be >= 1.0, but its"
                             " value is {}".format(self.l))
        if (self.gene_cutoff < 1):
            raise ValueError("'gene deletion cutoff' must be > 1.0, but its"
                             " value is {}".format(self.gene_cutoff))
        if (self.sample_cutoff < 1):
            raise ValueError("'sample deletion cutoff' must be > 1.0, but its"
                             " value is {}".format(self.sample_cutoff))
        if (self.chrom_cutoff < 1):
            raise ValueError("'chromosomes deletion cutoff' must be > 1.0, but"
                             " its value is {}".format(self.chrom_cutoff))
        if (self.mask_mode not in ['nan', 'random']):
            raise ValueError("'mask mode' must be either 'nan' or 'random',"
                             " but its value is {}".format(self.mask_mode))

    def _compute_MSR(self, chroms, genes, samples):
        """
        Computes the Mean Squared Residue (MSR) for the algorithm.

        Parameters
        ----------
        chroms : ndarray
            Contains 1 for a chromosome pair that belongs to the tricluster
            currently examined, 0 otherwise.
        genes : ndarray
            Contains 1 for a gene that belongs to the tricluster currently
            examined, 0 otherwise.
        samples : ndarray
            Contains 1 for a sample that belongs to the tricluster currently
            examined, 0 otherwise.

        Note
        ----
        Updates the n_chorms, n_genes, n_samples, MSR, MSR_chrom, MSR_gene and
        MSR_sample attributes.
        """
        chrom_idx = np.expand_dims(np.expand_dims(np.nonzero(chroms)[0], axis=1), axis=1)
        gene_idx = np.expand_dims(np.expand_dims(np.nonzero(genes)[0], axis=0), axis=2)
        sample_idx = np.expand_dims(np.expand_dims(np.nonzero(samples)[0], axis=0), axis=0)

        if (not chrom_idx.size) or (not gene_idx.size) or (not sample_idx.size):
            raise EmptyTriclusterException()

        subarr = self.D[chrom_idx, gene_idx, sample_idx]
        self.n_chroms = subarr.shape[0]
        self.n_genes = subarr.shape[1]
        self.n_samples = subarr.shape[2]

        with warnings.catch_warnings():  # We expect mean of NaNs here
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Computation of m_iJK
            m_iJK = np.nanmean(np.nanmean(subarr, axis=2), axis=1)
            m_iJK = np.expand_dims(np.expand_dims(m_iJK, axis=1), axis=1)

            # Computation of m_IjK
            m_IjK = np.nanmean(np.nanmean(subarr, axis=2), axis=0)
            m_IjK = np.expand_dims(np.expand_dims(m_IjK, axis=0), axis=2)

            # Computation of m_IJk
            m_IJk = np.nansum(np.nansum(subarr, axis=0, keepdims=1), axis=1, keepdims=1)
            m_IJk = m_IJk / ((subarr.shape[0] * subarr.shape[1]) - np.count_nonzero(np.isnan(subarr[:,:,0])))

            # Computation of m_IJK
            m_IJK = np.nanmean(subarr)

            # Computation of MSR
            residue = subarr - m_iJK - m_IjK - m_IJk + (2*m_IJK)
            SR = np.square(residue)

            self.MSR = np.nanmean(SR)
            self.MSR_chrom = np.nanmean(np.nanmean(SR, axis=2), axis=1)
            self.MSR_gene = np.nanmean(np.nanmean(SR, axis=2), axis=0)
            self.MSR_sample = np.nanmean(np.nanmean(SR, axis=0), axis=0)

            # Check tolerance
            self.MSR_chrom[self.MSR_chrom < self.tol] = 0
            self.MSR_gene[self.MSR_gene < self.tol] = 0
            self.MSR_sample[self.MSR_sample < self.tol] = 0
            self.MSR = 0 if (self.MSR < self.tol or np.isnan(self.MSR)) else self.MSR

    def _single_node_deletion(self, chroms, genes, samples):
        """
        The single node deletion routine of the algorithm.

        Parameters
        ----------
        chroms : ndarray
            Contains 1 for a chromosome pair that belongs to the tricluster
            currently examined, 0 otherwise.
        genes : ndarray
            Contains 1 for a gene that belongs to the tricluster currently
            examined, 0 otherwise.
        samples : ndarray
            Contains 1 for a sample that belongs to the tricluster currently
            examined, 0 otherwise.

        Returns
        -------
        chroms : ndarray
            Contains 1 for a chromosome pair that belongs to the tricluster
            examined, 0 otherwise.
        genes : ndarray
            Contains 1 for a gene that belongs to the tricluster examined,
            0 otherwise.
        samples : ndarray
            Contains 1 for a sample that belongs to the tricluster examined,
            0 otherwise.
        """
        self._compute_MSR(chroms, genes, samples)

        while (self.MSR > self.delta):
            chrom_idx = np.nanargmax(self.MSR_chrom)
            gene_idx = np.nanargmax(self.MSR_gene)
            sample_idx = np.nanargmax(self.MSR_sample)

            with warnings.catch_warnings():  # We expect mean of NaNs here
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if (self.MSR_chrom[chrom_idx] > self.MSR_gene[gene_idx]):
                    if (self.MSR_chrom[chrom_idx] > self.MSR_sample[sample_idx]):
                        # Delete chrom
                        nonz_idx = chroms.nonzero()[0]
                        chroms.put(nonz_idx[chrom_idx], 0)
                    else:
                        # Delete sample
                        nonz_idx = samples.nonzero()[0]
                        samples.put(nonz_idx[sample_idx], 0)
                else:
                    if (self.MSR_gene[gene_idx] > self.MSR_sample[sample_idx]):
                        # Delete gene
                        nonz_idx = genes.nonzero()[0]
                        genes.put(nonz_idx[gene_idx], 0)
                    else:
                        # Delete sample
                        nonz_idx = samples.nonzero()[0]
                        samples.put(nonz_idx[sample_idx], 0)

            self._compute_MSR(chroms, genes, samples)

        return chroms, genes, samples

    def _multiple_node_deletion(self, chroms, genes, samples):
        """
        The multiple node deletion routine of the algorithm.

        Parameters
        ----------
        chroms : ndarray
            Contains 1 for a chromosome pair that belongs to the tricluster
            currently examined, 0 otherwise.
        genes : ndarray
            Contains 1 for a gene that belongs to the tricluster currently
            examined, 0 otherwise.
        samples : ndarray
            Contains 1 for a sample that belongs to the tricluster currently
            examined, 0 otherwise.

        Returns
        -------
        chroms : ndarray
            Contains 1 for a chromosome pair that belongs to the tricluster
            examined, 0 otherwise.
        genes : ndarray
            Contains 1 for a gene that belongs to the tricluster examined,
            0 otherwise.
        samples : ndarray
            Contains 1 for a sample that belongs to the tricluster examined,
            0 otherwise.
        """
        self._compute_MSR(chroms, genes, samples)

        while (self.MSR > self.delta):
            deleted = 0

            with warnings.catch_warnings():  # We expect mean of NaNs here
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if (self.n_chroms > self.chrom_cutoff):
                    chroms_to_del = self.MSR_chrom > (self.l * self.MSR)
                    nonz_idx = chroms.nonzero()[0]
                    if (chroms_to_del.any()):
                        deleted = 1
                    chroms.put(nonz_idx[chroms_to_del], 0)

                if (self.n_genes > self.gene_cutoff):
                    genes_to_del = self.MSR_gene > (self.l * self.MSR)
                    nonz_idx = genes.nonzero()[0]
                    if (genes_to_del.any()):
                        deleted = 1
                    genes.put(nonz_idx[genes_to_del], 0)

                if (self.n_samples > self.sample_cutoff):
                    samples_to_del = self.MSR_sample > (self.l * self.MSR)
                    nonz_idx = samples.nonzero()[0]
                    if (samples_to_del.any()):
                        deleted = 1
                    samples.put(nonz_idx[samples_to_del], 0)

            if (not deleted):
                break

            self._compute_MSR(chroms, genes, samples)

        return chroms, genes, samples

    def _node_addition(self, chroms, genes, samples):
        """
        The single node addition routine of the algorithm.

        Parameters
        ----------
        chroms : ndarray
            Contains 1 for a chromosome pair that belongs to the tricluster
            currently examined, 0 otherwise.
        genes : ndarray
            Contains 1 for a gene that belongs to the tricluster currently
            examined, 0 otherwise.
        samples : ndarray
            Contains 1 for a sample that belongs to the tricluster currently
            examined, 0 otherwise.

        Returns
        -------
        chroms : ndarray
            Contains 1 for a chromosome pair that belongs to the tricluster
            examined, 0 otherwise.
        genes : ndarray
            Contains 1 for a gene that belongs to the tricluster examined,
            0 otherwise.
        samples : ndarray
            Contains 1 for a sample that belongs to the tricluster examined,
            0 otherwise.
        """

        while True:
            self._compute_MSR(chroms, genes, samples)
            n_chroms = np.count_nonzero(chroms)
            n_genes = np.count_nonzero(genes)
            n_samples = np.count_nonzero(samples)

            with warnings.catch_warnings():  # We expect mean of NaNs here
                warnings.simplefilter("ignore", category=RuntimeWarning)
                elems_to_add = self.MSR_chrom <= self.MSR
                nonz_idx = chroms.nonzero()[0]
                chroms.put(nonz_idx[elems_to_add], 1)

                elems_to_add = self.MSR_gene <= self.MSR
                nonz_idx = genes.nonzero()[0]
                genes.put(nonz_idx[elems_to_add], 1)

                elems_to_add = self.MSR_sample <= self.MSR
                nonz_idx = samples.nonzero()[0]
                samples.put(nonz_idx[elems_to_add], 1)

            if (n_chroms == np.count_nonzero(chroms)) and \
               (n_genes == np.count_nonzero(genes)) and \
               (n_samples == np.count_nonzero(samples)):
                break

        return chroms, genes, samples

    def _mask(self, chroms, genes, samples, minval, maxval):
        """
        Masks the values of the array that have been used in triclusters
        with either random float numbers, or nan.

        Parameters
        ----------
        chroms : ndarray
            Contains 1 for a chromosome pair that belongs to the tricluster
            currently examined, 0 otherwise.
        genes : ndarray
            Contains 1 for a gene that belongs to the tricluster currently
            examined, 0 otherwise.
        samples : ndarray
            Contains 1 for a sample that belongs to the tricluster currently
            examined, 0 otherwise.
        minval : float
            Lower boundary of the output interval for the random generator.
        maxval : float
            Upper boundary of the output interval for the random generator.
        """
        c = np.expand_dims(np.expand_dims(chroms.nonzero()[0], axis=1), axis=1)
        g = np.expand_dims(np.expand_dims(genes.nonzero()[0], axis=0), axis=2)
        s = np.expand_dims(np.expand_dims(samples.nonzero()[0], axis=0), axis=0)
        if (self.mask_mode == 'random'):
            shape = np.count_nonzero(chroms), np.count_nonzero(genes), np.count_nonzero(samples)
            mask_vals = np.random.uniform(minval, maxval, shape)
            self.D[c, g, s] = mask_vals
        else:
            self.D[c, g, s] = np.nan

    def fit(self, delta=2.5, l=1.005, chrom_cutoff=50, gene_cutoff=50,
            sample_cutoff=50, tol=1e-5, mask_mode='nan', verbose=False):
        """
        Runs the delta-TRIMAX algorithm with the given parameters.

        Parameters
        ----------
        delta : float, default 2.5
            The delta parameter of the algorithm. Must be > 0.0
        l : float, default 1.005
            The lambda parameter of the algorithm. Must be >= 1.0
        chrom_cutoff : int, default 50
            The deletion threshold for the chromosome axis
        gene_cutoff : int, default 50
            The deletion threshold for the gene axis
        sample_cutoff : int, default 50
            The deletion threshold for the sample axis
        tol : float, default 1e-5
            The algorithm's tolerance
        mask_mode : {'random', 'nan'}, default 'nan'
            The masking method for the clustered values. If 'random', the values
            are replaced by random floats. If 'nan', they are replaced by nan
            values.
        verbose : bool, default False
            Verbose mode for debugging.
        """
        self.delta = delta
        self.l = l
        self.chrom_cutoff = chrom_cutoff
        self.gene_cutoff = gene_cutoff
        self.sample_cutoff = sample_cutoff
        self.tol = tol
        self.mask_mode = mask_mode
        self._check_parameters()

        n_chroms, n_genes, n_samples = self.D.shape
        minval, maxval = np.nanmin(self.D), np.nanmax(self.D)

        result_chroms = []
        result_genes = []
        result_samples = []

        i = 1
        while True:
            if (verbose):
                print(i)
            chroms = np.ones(n_chroms, dtype=np.bool)
            genes = np.ones(n_genes, dtype=np.bool)
            samples = np.ones(n_samples, dtype=np.bool)

            # Multiple node deletion
            chroms, genes, samples = self._multiple_node_deletion(chroms,
                                                                  genes,
                                                                  samples)

            # Single node deletion
            chroms, genes, samples = self._single_node_deletion(chroms,
                                                                genes,
                                                                samples)

            # Node addition
            chroms, genes, samples = self._node_addition(chroms,
                                                         genes,
                                                         samples)

            # Check for trivial tricluster
            if (chroms.sum() == 1) or (genes.sum() == 1) or (samples.sum() == 1):
                break  # trivial bicluster
            # Check if the aren't any unused values in D
            if ((mask_mode == 'nan') and (np.isnan(self.D).all())):
                break

            # Mask values
            self._mask(chroms, genes, samples, minval, maxval)

            result_chroms.append(chroms)
            result_genes.append(genes)
            result_samples.append(samples)

            if (verbose):
                print("--- MSR = " + str(self.MSR))

            i += 1

        self.result_chroms = result_chroms
        self.result_genes = result_genes
        self.result_samples = result_samples

    def get_triclusters(self):
        """
        Returns the triclusters found by the algorithm.
        """
        return self.result_chroms, self.result_genes, self.result_samples
