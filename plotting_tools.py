# -*- coding: utf-8 -*-

from sympy import ceiling, floor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import seaborn as sns

import warnings


def plot_boxplot(data, xlabels=None, figsize=(8, 5), fontsize=16,
                 filename=None, dpi=150, paper_style=False,
                 figurename='boxplot'):
    """
    Creates a box plot to show the underlining distribution of the data.

    Parameters
    ----------
    data : ndarray
        The data to be plotted.
    xlabels : list of str, default None
        The labels of the x axis. If None given, numbers are plotted.
    figsize : tuple, default (8, 5)
        The size of the final plot.
    fontsize : int, default 16
        The size of the fonts in the plot
    filename : str, default None
        The path in which to save the resulting plot. The plot is saved as .png
    dpi : int, default 150
        The dpi quality of the final plot (Applies only when plot is saved in
        file)
    paper_style : bool, default False
        If True the plot produced is suitable for printing (with less colors
        and effects).
    figurename : str, default 'boxplot'
        The name of the plot.
    """
    fig = plt.figure(figurename, figsize=figsize)

    if (paper_style):
        sns.set_context("paper")
        sns.set_style("white")   # Make the background white

    plt.boxplot(data)

    plt.xlabel('samples', fontsize=fontsize, fontname='Times New Roman')
    plt.xticks(range(data.shape[2]+1), fontname='Times New Roman')
    if (xlabels is not None):
        plt.xticks(range(len(xlabels)+1), xlabels, rotation=90)
    plt.ylabel('gene expression', fontsize=fontsize, fontname='Times New Roman')
    plt.yticks(range(-5, 45, 5), fontname='Times New Roman')

    if (paper_style):
        sns.despine(fig)

    fig.tight_layout()
    if (filename is not None):
        plt.savefig(filename + '/' + figurename + '.png', dpi=dpi)
    fig.show()


def plot_array3D_slices(arr, start_slice=0, stop_slice=None, xlabels=None,
                        centroids=None, n_subplots=4, figsize=(17.075, 8.1125),
                        filename=None, dpi=150, fontsize=8, paper_style=False):
    '''
    Plots the horizontal slices from the given 3D array starting from
    start_slice and ending in stop_slice.

    Each window contains n_subplots slices.

    Parameters
    ----------
    arr : ndarray
        The data to be plotted. Must be 3-dimensional.
    start_slice : int, default 0
        The slice to start from.
    stop_slice : int, default None
        The slice until which to plot (not included).
    xlabels : list of str, default None
        The labels of the x axis. If None given, numbers are plotted.
    centroids : ndarray, default None
        The centroids of the slices.
    n_subplots : int, default 4
        The number of subplots in each window.
    figsize : tuple, default (8, 5)
        The size of the window.
    filename : str, default None
        The path in which to save the resulting plot. The plot is saved as .png
    dpi : int, default 150
        The dpi quality of the final plot (Applies only when plot is saved in
        file)
    fontsize : int, default 8
        The size of the fonts in the plot
    paper_style : bool, default False
        If True the plot produced is suitable for printing (with less colors
        and effects).
    '''

    if (stop_slice is None):
        stop_slice = arr.shape[1]
    n_slices = stop_slice - start_slice

    no_windows = (n_slices // n_subplots)
    if (n_slices % n_subplots > 0):
        no_windows += 1

    curr_slice = start_slice
    for win in range(1, no_windows+1):
        figurename = "From slice " + str(start_slice) + " to " + str(stop_slice) + " part " + str(win)
        fig = plt.figure(figurename, figsize=figsize)
        while (curr_slice - start_slice < n_slices) and (curr_slice - start_slice < win*n_subplots):
            if (n_subplots > 1):
                if ((n_slices // n_subplots) - win < 0):
                    plt.subplot(ceiling((n_slices % n_subplots) / 2.0), 2, ((curr_slice-start_slice) % n_subplots)+1)
                else:
                    plt.subplot(ceiling(n_subplots/2), 2, ((curr_slice-start_slice) % n_subplots)+1)

            if (paper_style):
                sns.set_context("paper")
                sns.set_style("white")   # Make the background white

            if (xlabels is not None):
                plt.xticks(range(arr.shape[2]),
                           xlabels,
                           rotation=25)
            plt.xticks(fontsize=fontsize, fontname='Times New Roman')
            plt.xlim(0, arr.shape[2]-1)

            plt.yticks(fontsize=fontsize, fontname='Times New Roman')

            if (paper_style):
                sns.set_context("paper")
                sns.set_style("white")   # Make the background white

            n_genes = 0
            for chrom in range(arr.shape[0]):  # for each gene in the slice
                plt.plot(range(arr.shape[2]), arr[chrom, curr_slice, :], linewidth=1)
                if (not np.isnan(arr[chrom, curr_slice, :]).all()):
                    n_genes += 1
            if (centroids is not None):
                plt.plot(range(arr.shape[2]), centroids[curr_slice, :],
                         color='k', linewidth=2)

            if (n_genes == 1):
                plt.title('Slice ' + str(curr_slice) + ' (' + str(n_genes) + ' gene)', fontname='Times New Roman', fontsize=fontsize+2)
            else:
                plt.title('Slice ' + str(curr_slice) + ' (' + str(n_genes) + ' genes)', fontname='Times New Roman', fontsize=fontsize+2)

            curr_slice += 1
            if (paper_style):
                sns.despine(fig)

        fig.tight_layout()
        if (filename is not None):
            plt.savefig(filename + '/' + figurename + '.png', dpi=dpi)
        fig.show()



def plot_clusters3D(n_clusters, arr, mode, labels, centroids=None,
                    xlabels=None, n_subplots=4, figsize=(17.075, 8.1125),
                    filename=None, dpi=150, fontsize=8, paper_style=False):
    '''
    Plots all clusters in multiple windows.
    Each window contains n_subplots clusters.

    Data array should be 3D.

    Parameters
    ----------
    n_clusters : int
        The number of clusters.
    arr : ndarray
        The data to be plotted.
    mode : (0, 1, 2)
        The axis along which the clustering was done.
    labels : ndarray
        The labels assigned to each datapoint during clustering.
    centroids : ndarray, default None
        The centroids of the slices.
    xlabels : list of str, default None
        The labels of the x axis. If None given, numbers are plotted.
    n_subplots : int, default 4
        The number of subplots in each window.
    figsize : tuple, default (17.075, 8.1125)
        The size of the window.
    filename : str, default None
        The path in which to save the resulting plot. The plot is saved as .png
    dpi : int, default 150
        The dpi quality of the final plot (Applies only when plot is saved in
        file)
    fontsize : int, default 8
        The size of the fonts in the plot
    paper_style : bool, default False
        If True the plot produced is suitable for printing (with less colors
        and effects).
    '''

    no_windows = (n_clusters // n_subplots)
    if (n_clusters % n_subplots > 0):
        no_windows += 1

    clust = 0
    for win in range(1, no_windows+1):
        figurename = "n_clusters=" + str(n_clusters) + "_" + str(win) + "_mode" + str(mode)
        fig = plt.figure(figurename, figsize=figsize)
        while (clust < n_clusters) and (clust < win*n_subplots):
            if (n_subplots > 1):
                if ((n_clusters // n_subplots) - win < 0):
                    plt.subplot(ceiling((n_clusters % n_subplots) / 2.0), 2, (clust % n_subplots)+1)
                else:
                    plt.subplot(ceiling(n_subplots/2), 2, (clust % n_subplots)+1)

            if (paper_style):
                sns.set_context("paper")
                sns.set_style("white")   # Make the background white

            cluster_genes = np.where(labels == clust)[0]

            if (len(cluster_genes) == 1):
                plt.title('Cluster ' + str(clust) + ' (' + str(len(cluster_genes)) + ' slice)', fontname='Times New Roman', fontsize=fontsize+2)
            else:
                plt.title('Cluster ' + str(clust) + ' (' + str(len(cluster_genes)) + ' slices)', fontname='Times New Roman', fontsize=fontsize+2)

            if (mode == 0) or (mode == 1):
                if (xlabels is not None):
                    plt.xticks(range(arr.shape[2]),
                                     xlabels,
                                     rotation=25)
                plt.xlim(0, arr.shape[2]-1)
            else:
                if (xlabels is not None):
                    plt.xticks(range(arr.shape[1]),
                                     xlabels,
                                     rotation=25)
                plt.xlim(0, arr.shape[1]-1)

            plt.xticks(fontsize=fontsize, fontname='Times New Roman')

            plt.yticks(fontsize=fontsize, fontname='Times New Roman')

            for gene_group in cluster_genes:  # for each gene in the cluster
                if (mode == 0):
                    for gene in arr[gene_group, :, :]:
                        plt.plot(range(arr.shape[2]), gene, linewidth=1)
                elif (mode == 1):
                    for gene in arr[:, gene_group, :]:
                        plt.plot(range(arr.shape[2]), gene, linewidth=1)
                else:
                    for sample in arr[:, :, gene_group]:
                        plt.plot(range(arr.shape[1]), sample, linewidth=1)
            if (centroids is not None):
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
            if (paper_style):
                sns.despine(fig)

        fig.tight_layout()       
        if (filename is not None):
            plt.savefig(filename + '/' + figurename + '.png', dpi=dpi)
    

def plot_clusters2D(n_clusters, arr, labels, centroids=None, xlabels=None,
                    n_subplots=4, figsize=(17.075, 8.1125), filename=None,
                    dpi=150, fontsize=8, paper_style=False):
    '''
    Plots all clusters in multiple windows.
    Each window contains n_subplots clusters.

    Data array should be 2D.

    Parameters
    ----------
    n_clusters : int
        The number of clusters.
    arr : ndarray
        The data to be plotted.
    labels : ndarray
        The labels assigned to each datapoint during clustering.
    centroids : ndarray, default None
        The centroids of the slices.
    xlabels : list of str, default None
        The labels of the x axis. If None given, numbers are plotted.
    n_subplots : int, default 4
        The number of subplots in each window.
    figsize : tuple, default (17.075, 8.1125)
        The size of the window.
    filename : str, default None
        The path in which to save the resulting plot. The plot is saved as .png
    dpi : int, default 150
        The dpi quality of the final plot (Applies only when plot is saved in
        file)
    fontsize : int, default 8
        The size of the fonts in the plot
    paper_style : bool, default False
        If True the plot produced is suitable for printing (with less colors
        and effects).
    '''
    no_windows = (n_clusters // n_subplots)
    if (n_clusters % n_subplots > 0):
        no_windows += 1

    clust = 0
    for win in range(1, no_windows+1):
        figurename = "n_clusters=" + str(n_clusters) + ",_" + str(win)
        fig = plt.figure(figurename, figsize=figsize)
        while (clust < n_clusters) and (clust < win*n_subplots):
            if (n_subplots > 1):
                if ((n_clusters // n_subplots) - win < 0):
                    plt.subplot(ceiling((n_clusters % n_subplots) / 2.0), 2, (clust % n_subplots)+1)
                else:
                    plt.subplot(ceiling(n_subplots/2), 2, (clust % n_subplots)+1)

            if (paper_style):
                sns.set_context("paper")
                sns.set_style("white")   # Make the background white

            cluster_genes = np.where(labels == clust)[0]

            if (len(cluster_genes == 1)):
                plt.title('Cluster ' + str(clust) + ' (' + str(len(cluster_genes)) + ' slice)', fontname='Times New Roman', fontsize=fontsize+2)
            else:
                plt.title('Cluster ' + str(clust) + ' (' + str(len(cluster_genes)) + ' slices)', fontname='Times New Roman', fontsize=fontsize+2)

            if (xlabels is not None):
                plt.xticks(range(arr.shape[1]),
                           xlabels,
                           rotation=25)
            plt.xlim(0, arr.shape[1]-1)
            plt.xticks(fontsize=fontsize, fontname='Times New Roman')
            plt.yticks(fontsize=fontsize, fontname='Times New Roman')

            for gene in cluster_genes:  # for each gene in the cluster
                plt.plot(range(arr.shape[1]), arr[gene, :], linewidth=1)
            if (centroids is not None):
                plt.plot(range(arr.shape[1]), centroids[clust, :], color='k',
                         linewidth=2)
            clust += 1
            if (paper_style):
                sns.despine(fig)

        fig.tight_layout()
        if (filename is not None):
            plt.savefig(filename + '/' + figurename + '.png', dpi=dpi)
        fig.show()


def plot_significant_clusters3D(n_clusters, arr, mode, labels, centroids=None,
                                xlabels=None, n_subplots=4,
                                figsize=(17.075, 8.1125), filename=None,
                                dpi=150, fontsize=8, paper_style=False):
    '''
    Plots only those clusters that contain more than 1 elements.
    Each window contains n_subplots clusters.

    Data array should be 3D.

    Parameters
    ----------
    n_clusters : int
        The number of clusters.
    arr : ndarray
        The data to be plotted.
    mode : (0, 1, 2)
        The axis along which the clustering was done.
    labels : ndarray
        The labels assigned to each datapoint during clustering.
    centroids : ndarray, default None
        The centroids of the slices.
    xlabels : list of str, default None
        The labels of the x axis. If None given, numbers are plotted.
    n_subplots : int, default 4
        The number of subplots in each window.
    figsize : tuple, default (17.075, 8.1125)
        The size of the window.
    filename : str, default None
        The path in which to save the resulting plot. The plot is saved as .png
    dpi : int, default 150
        The dpi quality of the final plot (Applies only when plot is saved in
        file)
    fontsize : int, default 8
        The size of the fonts in the plot
    paper_style : bool, default False
        If True the plot produced is suitable for printing (with less colors
        and effects).
    '''

    subplots = 1
    figurename = "Significant from n_clusters_" + str(n_clusters) + "_1_mode" + str(mode)
    fig = plt.figure(figurename, figsize=figsize)
    i = 2
    for clust in range(n_clusters):
        cluster_genes = np.where(labels == clust)[0]
        if (len(cluster_genes) > 1):
            if (n_subplots == 1):
                # We won't create any subplots
                if (subplots == 2):
                    # We have to open a new window
                    fig.tight_layout()
                    if (filename is not None):
                        plt.savefig(filename + '/' + figurename + '.png', dpi=dpi, bbox_inches='tight')
                    fig.show()
                    subplots = 1
                    figurename = "Significant from n_clusters_" + str(n_clusters) + "_" + str(i) + "_mode" + str(mode)
                    fig = plt.figure(figurename, figsize=figsize)
                    i += 1

                if (paper_style):
                    sns.set_context("paper")
                    sns.set_style("white")   # Make the background white

                plt.title('Cluster ' + str(clust) + ' (' + str(len(cluster_genes)) + ' slices)', fontname='Times New Roman', fontsize=fontsize+2)

                subplots += 1

                if (mode == 0) or (mode == 1):
                    if (xlabels is not None):
                        plt.xticks(range(arr.shape[2]),
                                   xlabels,
                                   rotation=25)
                    plt.xlim(0, arr.shape[2]-1)
                else:
                    if (xlabels is not None):
                        plt.xticks(range(arr.shape[1]),
                                   xlabels,
                                   rotation=25)
                    plt.xlim(0, arr.shape[1]-1)

                plt.xticks(fontsize=fontsize, fontname='Times New Roman')
                plt.yticks(fontsize=fontsize, fontname='Times New Roman')

                for gene_group in cluster_genes:  # for each gene in the cluster
                    if (mode == 0):
                        for gene in arr[gene_group, :, :]:
                            plt.plot(range(arr.shape[2]), gene, linewidth=1)
                    elif (mode == 1):
                        for gene in arr[:, gene_group, :]:
                            plt.plot(range(arr.shape[2]), gene, linewidth=1)
                    else:
                        for sample in arr[:, :, gene_group]:
                            plt.plot(range(arr.shape[1]), sample, linewidth=1)
                if (centroids is not None):
                    if (mode == 0):
                        plt.plot(range(arr.shape[2]), centroids[clust, :], color='k',
                                 linewidth=2)
                    elif (mode == 1):
                        plt.plot(range(arr.shape[2]), centroids[clust, :], color='k',
                                 linewidth=2)
                    else:
                        plt.plot(range(arr.shape[1]), centroids[clust, :], color='k',
                                 linewidth=2)
                if (paper_style):
                    sns.despine(fig)
            else:
                if (subplots == n_subplots+1):
                    fig.tight_layout()
                    if (filename is not None):
                        plt.savefig(filename + '/' + figurename + '.png', dpi=dpi, bbox_inches='tight')
                    fig.show()
                    subplots = 1
                    figurename = "Significant from n_clusters_" + str(n_clusters) + "_" + str(i) + "_mode" + str(mode)
                    fig = plt.figure(figurename, figsize=figsize)
                    i += 1
                plt.subplot(ceiling(n_subplots/2), 2, subplots)
                subplots += 1

                if (paper_style):
                    sns.set_context("paper")
                    sns.set_style("white")   # Make the background white

                plt.title('Cluster ' + str(clust) + ' (' + str(len(cluster_genes)) + ' slices)', fontname='Times New Roman', fontsize=fontsize+2)

                if (mode == 0) or (mode == 1):
                    if (xlabels is not None):
                        plt.xticks(range(arr.shape[2]),
                                   xlabels,
                                   rotation=25)
                    plt.xlim(0, arr.shape[2]-1)
                else:
                    if (xlabels is not None):
                        plt.xticks(range(arr.shape[1]),
                                   xlabels,
                                   rotation=25)
                    plt.xlim(0, arr.shape[1]-1)

                plt.xticks(fontsize=fontsize, fontname='Times New Roman')
                plt.yticks(fontsize=fontsize, fontname='Times New Roman')

                for gene_group in cluster_genes:  # for each gene in the cluster
                    if (mode == 0):
                        for gene in arr[gene_group, :, :]:
                            plt.plot(range(arr.shape[2]), gene, linewidth=1)
                    elif (mode == 1):
                        for gene in arr[:, gene_group, :]:
                            plt.plot(range(arr.shape[2]), gene, linewidth=1)
                    else:
                        for sample in arr[:, :, gene_group]:
                            plt.plot(range(arr.shape[1]), sample, linewidth=1)
                if (centroids is not None):
                    if (mode == 0):
                        plt.plot(range(arr.shape[2]), centroids[clust, :], color='k',
                                 linewidth=2)
                    elif (mode == 1):
                        plt.plot(range(arr.shape[2]), centroids[clust, :], color='k',
                                 linewidth=2)
                    else:
                        plt.plot(range(arr.shape[1]), centroids[clust, :], color='k',
                                 linewidth=2)
                if (paper_style):
                    sns.despine(fig)

    fig.tight_layout()
    if (filename is not None):
        plt.savefig(filename + '/' + figurename + '.png', dpi=dpi, bbox_inches='tight')
    fig.show()


def plot_significant_clusters2D(n_clusters, arr, labels, centroids=None,
                                xlabels=None, n_subplots=4,
                                figsize=(17.075, 8.1125), filename=None,
                                dpi=150, fontsize=8, paper_style=False):
    '''
    Plots only those clusters that contain more than 1 elements.
    Each window contains n_subplots clusters.

    Data array should be 2D.

    Parameters
    ----------
    n_clusters : int
        The number of clusters.
    arr : ndarray
        The data to be plotted.
    labels : ndarray
        The labels assigned to each datapoint during clustering.
    centroids : ndarray, default None
        The centroids of the slices.
    xlabels : list of str, default None
        The labels of the x axis. If None given, numbers are plotted.
    n_subplots : int, default 4
        The number of subplots in each window.
    figsize : tuple, default (17.075, 8.1125)
        The size of the window.
    filename : str, default None
        The path in which to save the resulting plot. The plot is saved as .png
    dpi : int, default 150
        The dpi quality of the final plot (Applies only when plot is saved in
        file)
    fontsize : int, default 8
        The size of the fonts in the plot
    paper_style : bool, default False
        If True the plot produced is suitable for printing (with less colors
        and effects).
    '''

    subplots = 1
    figurename = "Significant from n_clusters_" + str(n_clusters) + "_1"
    fig = plt.figure(figurename, figsize=figsize)
    i = 2
    for clust in range(n_clusters):
        cluster_genes = np.where(labels == clust)[0]
        if (len(cluster_genes) > 1):
            if (n_subplots == 1):
                # We won't create any subplots
                if (subplots == 2):
                    # We have to open a new window
                    fig.tight_layout()
                    if (filename is not None):
                        plt.savefig(filename + '/' + figurename + '.png', dpi=dpi, bbox_inches='tight')
                    fig.show()
                    subplots = 1
                    figurename = "Significant from n_clusters_" + str(n_clusters) + "_" + str(i)
                    fig = plt.figure(figurename, figsize=figsize)
                    i += 1
                plt.title('Cluster ' + str(clust) + ' (' + str(len(cluster_genes)) + ' slices)', fontname='Times New Roman', fontsize=fontsize+2)
                subplots += 1

                if (paper_style):
                    sns.set_context("paper")
                    sns.set_style("white")   # Make the background white

                if (xlabels is not None):
                    plt.xticks(range(arr.shape[1]),
                               xlabels,
                               rotation=25)

                plt.xlim(0, arr.shape[1]-1)
                plt.xticks(fontsize=fontsize, fontname='Times New Roman')
                plt.yticks(fontsize=fontsize, fontname='Times New Roman')

                for gene in cluster_genes:
                    plt.plot(range(arr.shape[1]), arr[gene, :], linewidth=1)
                if (centroids is not None):
                    plt.plot(range(arr.shape[1]), centroids[clust, :],
                             color='k', linewidth=2)
                if (paper_style):
                    sns.despine(fig)
            else:
                if (subplots == n_subplots+1):
                    fig.tight_layout()
                    if (filename is not None):
                        plt.savefig(filename + '/' + figurename + '.png', dpi=dpi)
                    fig.show()
                    subplots = 1
                    figurename = "Significant from n_clusters_" + str(n_clusters) + "_" + str(i)
                    fig = plt.figure(figurename, figsize=figsize)
                    i += 1
                plt.subplot(ceiling(n_subplots/2), 2, subplots)
                subplots += 1

                if (paper_style):
                    sns.set_context("paper")
                    sns.set_style("white")   # Make the background white

                plt.title('Cluster ' + str(clust) + ' (' + str(len(cluster_genes)) + ' slices)', fontname='Times New Roman', fontsize=fontsize+2)
                if (xlabels is not None):
                    plt.xticks(range(arr.shape[1]),
                               xlabels,
                               rotation=25)

                plt.xlim(0, arr.shape[1]-1)
                plt.xticks(fontsize=fontsize, fontname='Times New Roman')
                plt.yticks(fontsize=fontsize, fontname='Times New Roman')

                for gene in cluster_genes:
                    plt.plot(range(arr.shape[1]), arr[gene, :], linewidth=1)
                if (centroids is not None):
                    plt.plot(range(arr.shape[1]), centroids[clust, :],
                             color='k', linewidth=2)
                if (paper_style):
                    sns.despine(fig)

    fig.tight_layout()
    if (filename is not None):
        plt.savefig(filename + '/' + figurename + '.png', dpi=dpi)
    fig.show()


def plot_centroids2D(n_clusters, arr, labels, xlabels=None, n_subplots=4,
                     error_bars='sd', figsize=(17.075, 8.1125), filename=None,
                     dpi=150, fontsize=8, paper_style=False):
    '''
    Plots only the centroids of the clusters with error bars.

    Data array should be 2D.

    Parameters
    ----------
    n_clusters : int
        The number of clusters.
    arr : ndarray
        The data to be plotted.
    labels : ndarray
        The labels assigned to each datapoint during clustering.
    xlabels : list of str, default None
        The labels of the x axis. If None given, numbers are plotted.
    n_subplots : int, default 4
        The number of subplots in each window.
    error_bars : {'sd', 'range'}
        The type of the error bars to be plotted.
    figsize : tuple, default (17.075, 8.1125)
        The size of the window.
    filename : str, default None
        The path in which to save the resulting plot. The plot is saved as .png
    dpi : int, default 150
        The dpi quality of the final plot (Applies only when plot is saved in
        file)
    fontsize : int, default 8
        The size of the fonts in the plot
    paper_style : bool, default False
        If True the plot produced is suitable for printing (with less colors
        and effects).
    '''

    ymax = ceiling(np.nanmax(arr))

    no_windows = (n_clusters // n_subplots)
    if (n_clusters % n_subplots > 0):
        no_windows += 1

    clust = 0
    with warnings.catch_warnings():  # We expect mean of NaNs here
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for win in range(1, no_windows+1):
            figurename = "Centroids of n_clusters=" + str(n_clusters) + "_" + str(win) + '_' + error_bars
            fig = plt.figure(figurename, figsize=figsize)
            while (clust < n_clusters) and (clust < win*n_subplots):
                if (n_subplots > 1):
                    if ((n_clusters // n_subplots) - win < 0):
                        plt.subplot(ceiling((n_clusters % n_subplots) / 2.0), 2, (clust % n_subplots)+1)
                    else:
                        plt.subplot(ceiling(n_subplots/2), 2, (clust % n_subplots)+1)

                if (paper_style):
                    sns.set_context("paper")
                    sns.set_style("white")   # Make the background white

                plt.title('Cluster ' + str(clust), fontname='Times New Roman', fontsize=fontsize+2)
                if (xlabels is not None):
                    plt.xticks(range(arr.shape[1]),
                               xlabels,
                               rotation=25)
                plt.xlim(0, arr.shape[1]-1)
                plt.xticks(fontsize=fontsize, fontname='Times New Roman')

                plt.yticks(range(ymax), fontsize=fontsize, fontname='Times New Roman')

                # Find gene indices of the current cluster
                cluster_genes = np.where(labels == clust)[0]
                # Compute the centroid
                centroid = np.nanmean(arr[cluster_genes, :], axis=0)

                # Compute the errors for each sample
                if (error_bars == 'range'):
                    divergence = arr[cluster_genes, :] - centroid
                    errors = []
                    errors.append(np.absolute(np.nanmin(divergence, axis=0)))
                    errors.append(np.absolute(np.nanmax(divergence, axis=0)))
                else:
                    errors = np.nanstd(arr[cluster_genes, :])

                # Plot the centroids with error bars
                plt.errorbar(range(arr.shape[1]), centroid, yerr=errors)

                clust += 1
                if (paper_style):
                    sns.despine(fig)

            fig.tight_layout()
            if (filename is not None):
                plt.savefig(filename + '/' + figurename + '.png', dpi=dpi)
            fig.show()


def plot_centroids3D(n_clusters, arr, mode, labels, xlabels=None,
                     n_subplots=4, error_bars='range',
                     figsize=(17.075, 8.1125), filename=None, dpi=150,
                     fontsize=8, paper_style=False):
    '''
    Plots only the centroids of the clusters with error bars.

    Data array should be 3D.

    Parameters
    ----------
    n_clusters : int
        The number of clusters.
    arr : ndarray
        The data to be plotted.
    mode : (0, 1, 2)
        The axis along which the clustering was done.
    labels : ndarray
        The labels assigned to each datapoint during clustering.
    xlabels : list of str, default None
        The labels of the x axis. If None given, numbers are plotted.
    n_subplots : int, default 4
        The number of subplots in each window.
    error_bars : {'sd', 'range'}
        The type of the error bars to be plotted.
    figsize : tuple, default (17.075, 8.1125)
        The size of the window.
    filename : str, default None
        The path in which to save the resulting plot. The plot is saved as .png
    dpi : int, default 150
        The dpi quality of the final plot (Applies only when plot is saved in
        file)
    fontsize : int, default 8
        The size of the fonts in the plot
    paper_style : bool, default False
        If True the plot produced is suitable for printing (with less colors
        and effects).
    '''

    ymax = ceiling(np.nanmax(arr))

    no_windows = (n_clusters // n_subplots)
    if (n_clusters % n_subplots > 0):
        no_windows += 1

    clust = 0
    with warnings.catch_warnings():  # We expect mean of NaNs here
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for win in range(1, no_windows+1):
            figurename = "Centroids of n_clusters=" + str(n_clusters) + "_" + str(win) + "_mode" + str(mode) + '_' + error_bars
            fig = plt.figure(figurename, figsize=figsize)
            while (clust < n_clusters) and (clust < win*n_subplots):
                if (n_subplots > 1):
                    if ((n_clusters // n_subplots) - win < 0):
                        plt.subplot(ceiling((n_clusters % n_subplots) / 2.0), 2, (clust % n_subplots)+1)
                    else:
                        plt.subplot(ceiling(n_subplots/2), 2, (clust % n_subplots)+1)

                if (paper_style):
                    sns.set_context("paper")
                    sns.set_style("white")   # Make the background white

                plt.title('Cluster ' + str(clust), fontname='Times New Roman', fontsize=fontsize+2)

                if (mode == 0) or (mode == 1):
                    if (xlabels is not None):
                        plt.xticks(range(arr.shape[2]),
                                   xlabels,
                                   rotation=25)
                    plt.xlim(0, arr.shape[2]-1)
                else:
                    if (xlabels is not None):
                        plt.xticks(range(arr.shape[1]),
                                   xlabels,
                                   rotation=25)
                    plt.xlim(0, arr.shape[1]-1)

                plt.xticks(fontsize=fontsize, fontname='Times New Roman')

                plt.yticks(range(ymax), fontsize=fontsize, fontname='Times New Roman')

                # Find gene indices of the current cluster
                cluster_genes = np.where(labels == clust)[0]

                # Compute the centroid and the errors for each sample
                # and plot the results
                if (mode == 0):
                    centroid = np.nanmean(np.nanmean(arr[cluster_genes, :, :], axis=0), axis=0)
                    if (error_bars == 'range'):
                        divergence = arr[cluster_genes, :, :] - np.broadcast_to(centroid, (len(cluster_genes), arr.shape[1], arr.shape[2]))
                        errors = []
                        errors.append(np.absolute(np.nanmin(np.nanmin(divergence, axis=0), axis=0)))
                        errors.append(np.absolute(np.nanmax(np.nanmax(divergence, axis=0), axis=0)))
                    else:
                        errors = np.nanstd(np.nanstd(arr[cluster_genes, :, :], axis=1), axis=0)
                    plt.errorbar(range(arr.shape[2]), centroid, yerr=errors)
                elif (mode == 1):
                    centroid = np.nanmean(np.nanmean(arr[:, cluster_genes, :], axis=1), axis=0)
                    if (error_bars == 'range'):
                        divergence = arr[:, cluster_genes, :] - np.broadcast_to(centroid, (arr.shape[0], len(cluster_genes), arr.shape[2]))
                        errors = []
                        errors.append(np.absolute(np.nanmin(np.nanmin(divergence, axis=1), axis=0)))
                        errors.append(np.absolute(np.nanmax(np.nanmax(divergence, axis=1), axis=0)))
                    else:
                        errors = np.nanstd(np.nanstd(arr[:, cluster_genes, :], axis=1), axis=0)
                    plt.errorbar(range(arr.shape[2]), centroid, yerr=errors)
                else:
                    centroid = np.nanmean(np.nanmean(arr[:, :, cluster_genes], axis=2), axis=0)
                    if (error_bars == 'range'):
                        divergence = arr[:, :, cluster_genes] - np.expand_dims(np.expand_dims(centroid, axis=0), axis=2)
                        errors = []
                        errors.append(np.absolute(np.nanmin(np.nanmin(divergence, axis=2), axis=0)))
                        errors.append(np.absolute(np.nanmax(np.nanmax(divergence, axis=2), axis=0)))
                    else:
                        errors = np.nanstd(np.nanstd(arr[:, :, cluster_genes], axis=2), axis=0)
                    plt.errorbar(range(arr.shape[1]), centroid, yerr=errors)

                clust += 1
                if (paper_style):
                    sns.despine(fig)

            fig.tight_layout()
            if (filename is not None):
                plt.savefig(filename + '/' + figurename + '.png', dpi=dpi)
            fig.show()


def plot_significant_centroids2D(n_clusters, arr, labels, xlabels=None,
                                 n_subplots=4, error_bars='range',
                                 figsize=(17.075, 8.1125), filename=None,
                                 dpi=150, fontsize=8, paper_style=False):
    '''
    Plots only the centroids of the significant clusters with error bars.

    Data array should be 2D.

    Parameters
    ----------
    n_clusters : int
        The number of clusters.
    arr : ndarray
        The data to be plotted.
    labels : ndarray
        The labels assigned to each datapoint during clustering.
    xlabels : list of str, default None
        The labels of the x axis. If None given, numbers are plotted.
    n_subplots : int, default 4
        The number of subplots in each window.
    error_bars : {'sd', 'range'}
        The type of the error bars to be plotted.
    figsize : tuple, default (17.075, 8.1125)
        The size of the window.
    filename : str, default None
        The path in which to save the resulting plot. The plot is saved as .png
    dpi : int, default 150
        The dpi quality of the final plot (Applies only when plot is saved in
        file)
    fontsize : int, default 8
        The size of the fonts in the plot
    paper_style : bool, default False
        If True the plot produced is suitable for printing (with less colors
        and effects).
    '''

    ymax = ceiling(np.nanmax(arr))

    subplots = 1
    figurename = "Significant centroids from n_clusters_" + str(n_clusters) + "_1_" + error_bars
    fig = plt.figure(figurename, figsize=figsize)
    i = 2
    with warnings.catch_warnings():  # We expect mean of NaNs here
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for clust in range(n_clusters):
            cluster_genes = np.where(labels == clust)[0]
            if (len(cluster_genes) > 1):
                if (n_subplots == 1):
                    # We won't create any subplots
                    if (subplots == 2):
                        # We have to open a new window
                        fig.tight_layout()
                        if (filename is not None):
                            plt.savefig(filename + '/' + figurename + '.png', dpi=dpi, bbox_inches='tight')
                        fig.show()
                        subplots = 1
                        figurename = "Significant centroids from n_clusters_" + str(n_clusters) + "_" + str(i) + '_' + error_bars
                        fig = plt.figure(figurename, figsize=figsize)
                        i += 1
                    plt.title('Cluster ' + str(clust) + ' (' + str(len(cluster_genes)) + ' slices)', fontname='Times New Roman', fontsize=fontsize+2)
                    subplots += 1

                    if (paper_style):
                        sns.set_context("paper")
                        sns.set_style("white")   # Make the background white

                    if (xlabels is not None):
                        plt.xticks(range(arr.shape[1]),
                                   xlabels,
                                   rotation=25)

                    plt.xticks(fontsize=fontsize, fontname='Times New Roman')
                    plt.xlim(0, arr.shape[1]-1)

                    plt.yticks(range(ymax), fontsize=fontsize, fontname='Times New Roman')
    
                    # Compute the centroid and the errors for each sample
                    # and plot the results
                    # Compute the centroid
                    centroid = np.nanmean(arr[cluster_genes, :], axis=0)

                    # Compute the errors for each sample
                    if (error_bars == 'range'):
                        divergence = arr[cluster_genes, :] - centroid
                        errors = []
                        errors.append(np.absolute(np.nanmin(divergence, axis=0)))
                        errors.append(np.absolute(np.nanmax(divergence, axis=0)))
                    else:
                        errors = np.nanstd(arr[cluster_genes, :])
                    # Plot the centroids with error bars
                    plt.errorbar(range(arr.shape[1]), centroid, yerr=errors)

                    if (paper_style):
                        sns.despine(fig)
                else:
                    if (subplots == n_subplots+1):
                        fig.tight_layout()
                        if (filename is not None):
                            plt.savefig(filename + '/' + figurename + '.png', dpi=dpi)
                        fig.show()
                        subplots = 1
                        figurename = "Significant centroids from n_clusters_" + str(n_clusters) + "_" + str(i) + '_' + error_bars
                        fig = plt.figure(figurename, figsize=figsize)
                        i += 1
                    plt.subplot(ceiling(n_subplots/2), 2, subplots)
                    subplots += 1

                    if (paper_style):
                        sns.set_context("paper")
                        sns.set_style("white")   # Make the background white

                    plt.title('Cluster ' + str(clust) + ' (' + str(len(cluster_genes)) + ' slices)', fontname='Times New Roman', fontsize=fontsize+2)
                    if (xlabels is not None):
                        plt.xticks(range(arr.shape[1]),
                                   xlabels,
                                   rotation=25)

                    plt.xlim(0, arr.shape[1]-1)
                    plt.xticks(fontsize=fontsize, fontname='Times New Roman')

                    plt.yticks(range(ymax), fontsize=fontsize, fontname='Times New Roman')

                    # Compute the centroid and the errors for each sample
                    # and plot the results
                    # Compute the centroid
                    centroid = np.nanmean(arr[cluster_genes, :], axis=0)

                    # Compute the errors for each sample
                    if (error_bars == 'range'):
                        divergence = arr[cluster_genes, :] - centroid
                        errors = []
                        errors.append(np.absolute(np.nanmin(divergence, axis=0)))
                        errors.append(np.absolute(np.nanmax(divergence, axis=0)))
                    else:
                        errors = np.nanstd(arr[cluster_genes, :])
                    # Plot the centroids with error bars
                    plt.errorbar(range(arr.shape[1]), centroid, yerr=errors)

                    if (paper_style):
                        sns.despine(fig)

    fig.tight_layout()
    if (filename is not None):
        plt.savefig(filename + '/' + figurename + '.png', dpi=dpi)
    fig.show()


def plot_significant_centroids3D(n_clusters, arr, mode, labels, xlabels=None,
                                 n_subplots=4, error_bars='range',
                                 figsize=(17.075, 8.1125), filename=None,
                                 dpi=150, fontsize=8, paper_style=False):
    '''
    Plots only the centroids of the significant clusters with error bars.

    Data array should be 3D.

    Parameters
    ----------
    n_clusters : int
        The number of clusters.
    arr : ndarray
        The data to be plotted.
    mode : (0, 1, 2)
        The axis along which the clustering was done.
    labels : ndarray
        The labels assigned to each datapoint during clustering.
    xlabels : list of str, default None
        The labels of the x axis. If None given, numbers are plotted.
    n_subplots : int, default 4
        The number of subplots in each window.
    error_bars : {'sd', 'range'}
        The type of the error bars to be plotted.
    figsize : tuple, default (17.075, 8.1125)
        The size of the window.
    filename : str, default None
        The path in which to save the resulting plot. The plot is saved as .png
    dpi : int, default 150
        The dpi quality of the final plot (Applies only when plot is saved in
        file)
    fontsize : int, default 8
        The size of the fonts in the plot
    paper_style : bool, default False
        If True the plot produced is suitable for printing (with less colors
        and effects).
    '''

    ymax = int(ceiling(np.nanmax(arr)))
    ymin = int(floor(np.nanmin(arr)))

    subplots = 1
    figurename = "Significant centroids from n_clusters_" + str(n_clusters) + "_1_mode" + str(mode) + '_' + error_bars
    fig = plt.figure(figurename, figsize=figsize)
    i = 2
    with warnings.catch_warnings():  # We expect mean of NaNs here
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for clust in range(n_clusters):
            cluster_genes = np.where(labels == clust)[0]
            if (len(cluster_genes) > 1):
                if (n_subplots == 1):
                    # We won't create any subplots
                    if (subplots == 2):
                        # We have to open a new window
                        fig.tight_layout()
                        if (filename is not None):
                            plt.savefig(filename + '/' + figurename + '.png', dpi=dpi, bbox_inches='tight')
                        fig.show()
                        subplots = 1
                        figurename = "Significant centroids from n_clusters_" + str(n_clusters) + "_" + str(i) + "_mode" + str(mode) + '_' + error_bars
                        fig = plt.figure(figurename, figsize=figsize)
                        i += 1
                    plt.title('Cluster ' + str(clust) + ' (' + str(len(cluster_genes)) + ' slices)', fontname='Times New Roman', fontsize=fontsize+2)
                    subplots += 1

                    if (paper_style):
                        sns.set_context("paper")
                        sns.set_style("white")   # Make the background white
        
                    if (mode == 0) or (mode == 1):
                        if (xlabels is not None):
                            plt.xticks(range(arr.shape[2]),
                                       xlabels,
                                       rotation=25)
                        plt.xlim(0, arr.shape[2]-1)
                    else:
                        if (xlabels is not None):
                            plt.xticks(range(arr.shape[1]),
                                       xlabels,
                                       rotation=25)
                        plt.xlim(0, arr.shape[1]-1)

                    plt.xticks(fontsize=fontsize, fontname='Times New Roman')
                    plt.yticks(fontsize=fontsize, fontname='Times New Roman')
                    plt.ylim(ymin, ymax)

                    # Compute the centroid and the errors for each sample
                    # and plot the results
                    if (mode == 0):
                        centroid = np.nanmean(np.nanmean(arr[cluster_genes, :, :], axis=0), axis=0)
                        if (error_bars == 'range'):
                            divergence = arr[cluster_genes, :, :] - np.broadcast_to(centroid, (len(cluster_genes), arr.shape[1], arr.shape[2]))
                            errors = []
                            errors.append(np.absolute(np.nanmin(np.nanmin(divergence, axis=0), axis=0)))
                            errors.append(np.absolute(np.nanmax(np.nanmax(divergence, axis=0), axis=0)))
                        else:
                            errors = np.nanstd(np.nanstd(arr[cluster_genes, :, :], axis=1), axis=0)
                        plt.errorbar(range(arr.shape[2]), centroid, yerr=errors)
                    elif (mode == 1):
                        centroid = np.nanmean(np.nanmean(arr[:, cluster_genes, :], axis=1), axis=0)
                        if (error_bars == 'range'):
                            divergence = arr[:, cluster_genes, :] - np.broadcast_to(centroid, (arr.shape[0], len(cluster_genes), arr.shape[2]))
                            errors = []
                            errors.append(np.absolute(np.nanmin(np.nanmin(divergence, axis=1), axis=0)))
                            errors.append(np.absolute(np.nanmax(np.nanmax(divergence, axis=1), axis=0)))
                        else:
                            errors = np.nanstd(np.nanstd(arr[:, cluster_genes, :], axis=1), axis=0)
                        plt.errorbar(range(arr.shape[2]), centroid, yerr=errors)
                    else:
                        centroid = np.nanmean(np.nanmean(arr[:, :, cluster_genes], axis=2), axis=0)
                        if (error_bars == 'range'):
                            divergence = arr[:, :, cluster_genes] - np.expand_dims(np.expand_dims(centroid, axis=0), axis=2)
                            errors = []
                            errors.append(np.absolute(np.nanmin(np.nanmin(divergence, axis=2), axis=0)))
                            errors.append(np.absolute(np.nanmax(np.nanmax(divergence, axis=2), axis=0)))
                        else:
                            errors = np.nanstd(np.nanstd(arr[:, :, cluster_genes], axis=2), axis=0)
                        plt.errorbar(range(arr.shape[1]), centroid, yerr=errors)

                    if (paper_style):
                        sns.despine(fig)
                else:
                    if (subplots == n_subplots+1):
                        fig.tight_layout()
                        if (filename is not None):
                            plt.savefig(filename + '/' + figurename + '.png', dpi=dpi, bbox_inches='tight')
                        fig.show()
                        subplots = 1
                        figurename = "Significant centroids from n_clusters_" + str(n_clusters) + "_" + str(i) + "_mode" + str(mode) + '_' + error_bars
                        fig = plt.figure(figurename, figsize=figsize)
                        i += 1
                    plt.subplot(ceiling(n_subplots/2), 2, subplots)
                    subplots += 1

                    if (paper_style):
                        sns.set_context("paper")
                        sns.set_style("white")   # Make the background white

                    plt.title('Cluster ' + str(clust) + ' (' + str(len(cluster_genes)) + ' slices)', fontname='Times New Roman', fontsize=fontsize+2)

                    if (mode == 0) or (mode == 1):
                        if (xlabels is not None):
                            plt.xticks(range(arr.shape[2]),
                                       xlabels,
                                       rotation=25)
                        plt.xlim(0, arr.shape[2]-1)
                    else:
                        if (xlabels is not None):
                            plt.xticks(range(arr.shape[1]),
                                       xlabels,
                                       rotation=25)
                        plt.xlim(0, arr.shape[1]-1)

                    plt.xticks(fontsize=fontsize, fontname='Times New Roman')
                    plt.yticks(fontsize=fontsize, fontname='Times New Roman')
                    plt.ylim(ymin, ymax)

                    # Compute the centroid and the errors for each sample
                    # and plot the results
                    if (mode == 0):
                        centroid = np.nanmean(np.nanmean(arr[cluster_genes, :, :], axis=0), axis=0)
                        if (error_bars == 'range'):
                            divergence = arr[cluster_genes, :, :] - np.broadcast_to(centroid, (len(cluster_genes), arr.shape[1], arr.shape[2]))
                            errors = []
                            errors.append(np.absolute(np.nanmin(np.nanmin(divergence, axis=0), axis=0)))
                            errors.append(np.absolute(np.nanmax(np.nanmax(divergence, axis=0), axis=0)))
                        else:
                            errors = np.nanstd(np.nanstd(arr[cluster_genes, :, :], axis=1), axis=0)
                        plt.errorbar(range(arr.shape[2]), centroid, yerr=errors)
                    elif (mode == 1):
                        centroid = np.nanmean(np.nanmean(arr[:, cluster_genes, :], axis=1), axis=0)
                        if (error_bars == 'range'):
                            divergence = arr[:, cluster_genes, :] - np.broadcast_to(centroid, (arr.shape[0], len(cluster_genes), arr.shape[2]))
                            errors = []
                            errors.append(np.absolute(np.nanmin(np.nanmin(divergence, axis=1), axis=0)))
                            errors.append(np.absolute(np.nanmax(np.nanmax(divergence, axis=1), axis=0)))
                        else:
                            errors = np.nanstd(np.nanstd(arr[:, cluster_genes, :], axis=1), axis=0)
                        plt.errorbar(range(arr.shape[2]), centroid, yerr=errors)
                    else:
                        centroid = np.nanmean(np.nanmean(arr[:, :, cluster_genes], axis=2), axis=0)
                        if (error_bars == 'range'):
                            divergence = arr[:, :, cluster_genes] - np.expand_dims(np.expand_dims(centroid, axis=0), axis=2)
                            errors = []
                            errors.append(np.absolute(np.nanmin(np.nanmin(divergence, axis=2), axis=0)))
                            errors.append(np.absolute(np.nanmax(np.nanmax(divergence, axis=2), axis=0)))
                        else:
                            errors = np.nanstd(np.nanstd(arr[:, :, cluster_genes], axis=2), axis=0)
                        plt.errorbar(range(arr.shape[1]), centroid, yerr=errors)

                    if (paper_style):
                        sns.despine(fig)

    fig.tight_layout()
    if (filename is not None):
        plt.savefig(filename + '/' + figurename + '.png', dpi=dpi, bbox_inches='tight')
    fig.show()


def nearest_neighbors(A, start):
    """Nearest neighbor algorithm.
    A is an NxN correlation matrix of N elements.
    start is the index of the starting location.
    Returns the path and cost of the found solution.

    This function regards matrix A as a NxN matrix of the distances
    between N nodes in a graph. Then it computes the maximum weight
    path in the graph from the start index.

    Parameters
    ----------
    A : ndarray
        The data (a NxN correlation matrix of N elements).
    start : int
        The index of the starting location for the NN algorithm.
    """
    path = [start]
    cost = 0
    N = A.shape[0]
    mask = np.ones(N, dtype=bool)   # boolean values indicating which
                                    # locations have not been visited
    mask[start] = False

    for i in range(N-1):
        last = path[-1]

        # find minimum of remaining locations
        next_ind = np.argmax(A[last][mask])

        # convert to original location
        next_loc = np.arange(N)[mask][next_ind]

        path.append(next_loc)
        mask[next_loc] = False
        cost += A[last, next_loc]

    return path, cost


def plot_surface(n_clusters, centroids):
    """
    Plots a 3D surface from the centroids of the clusters.
    Before the plotting, it finds a proper ordering of the centoids,
    ie with compact hills and valleys.

    Parameters
    ----------
    n_clusters : int
        The number of the clusters.
    centroids : ndarray
        The centroids of the clusters.
    """
    # Compute the pearson's correlation coefficient for every centroid pair
    pearson = np.corrcoef(centroids)

    # Run repetitive nearest neighbor algorithm in order to find
    # a sorting of the chromosomes by their correlation
    max_cost = ([], 0)
    for start in range(n_clusters):
        nn_sol = nearest_neighbors(pearson, start)
        if (nn_sol[1] > max_cost[1]):
            max_cost = nn_sol

    # Best sorting is contained in max_cost[0]
    best_sol = max_cost[0]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    centroids_ordered = []
    for clust in best_sol:
        centroids_ordered.append(centroids[clust, :])
    centroids_ordered = np.asarray(centroids_ordered)

    X = range(centroids.shape[1])
    Y = range(n_clusters)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, centroids_ordered)

    ax.set_xlabel('samples')
    ax.set_ylabel('clusters')
    ax.set_zlabel('gene expression')


def plot_contour(n_clusters, centroids, mode, xlabels=None, plot_before=False,
                 figsize=(17.075, 8.1125), filename=None, dpi=150,
                 fontsize=8):
    """
    Creates a contour plot from the centroids of the clusters. Before the
    plotting, it finds a proper ordering of the centoids,
    ie with compact hills and valleys.

    Parameters
    ----------
    n_clusters : int
        The number of clusters.
    centroids : ndarray
        The centroids of the clusters.
    mode : (0, 1, 2)
        The axis along which the clustering was done.
    xlabels : list of str, default None
        The labels of the x axis. If None given, numbers are plotted.
    plot_before : bool, default False
        If True, it also prints the contour plot before the NN algorithm.
    figsize : tuple, default (17.075, 8.1125)
        The size of the window.
    filename : str, default None
        The path in which to save the resulting plot. The plot is saved as .png
    dpi : int, default 150
        The dpi quality of the final plot (Applies only when plot is saved in
        file)
    fontsize : int, default 8
        The size of the fonts in the plot
    """
    if (plot_before):
        figurename = "Contour plot for n_clusters=" + str(n_clusters) + " before ordering"
        fig = plt.figure(figurename, figsize=figsize)
        X = range(centroids.shape[1])
        Y = range(n_clusters)
        X, Y = np.meshgrid(X, Y)
        cset = plt.contourf(X, Y, centroids, cmap="rainbow")
        plt.colorbar(cset)
        # Also cmap="viridis" or cmap=plt.cm.jet look nice

        if (mode == 0) or (mode == 1):
            plt.xlabel('samples')
        else:
            plt.xlabel('genes')
        plt.ylabel('clusters')

        if (xlabels is not None):
            plt.xticks(range(centroids.shape[1]), xlabels, fontsize=fontsize,
                       rotation=25)
        else:
            plt.xticks(range(centroids.shape[1]),
                       range(1, centroids.shape[1]+1), fontsize=fontsize)

        if (n_clusters <= 100):
            plt.yticks(range(n_clusters), range(n_clusters), fontsize=fontsize)
        else:
            plt.yticks(range(n_clusters), ["-"] * n_clusters, fontize=fontsize)

        fig.tight_layout()
        if (filename is not None):
            plt.savefig(filename + '/' + figurename + '.png', dpi=dpi)
        fig.show()

    # Compute the pearson's correlation coefficient for every centroid pair
    pearson = np.corrcoef(centroids)

    # Run repetitive nearest neighbor algorithm in order to find
    # a sorting of the chromosomes by their correlation
    max_cost = nearest_neighbors(pearson, 0)
    for start in range(1, n_clusters):
        nn_sol = nearest_neighbors(pearson, start)
        if (nn_sol[1] > max_cost[1]):
            max_cost = nn_sol

    # Best sorting is contained in max_cost[0]
    best_sol = max_cost[0]

    centroids_ordered = []
    for clust in best_sol:
        centroids_ordered.append(centroids[clust, :])
    centroids_ordered = np.asarray(centroids_ordered)

    X = range(centroids.shape[1])
    Y = range(n_clusters)
    X, Y = np.meshgrid(X, Y)

    figurename = "Contour plot for n_clusters=" + str(n_clusters) + " after proper ordering"
    fig = plt.figure(figurename, figsize=figsize)
    cset = plt.contourf(X, Y, centroids_ordered, cmap="rainbow")
    plt.colorbar(cset)
    # Also cmap="viridis" or cmap=plt.cm.jet look nice

    if (mode == 0) or (mode == 1):
        plt.xlabel('samples')
    else:
        plt.xlabel('genes')
    plt.ylabel('clusters')

    if (xlabels is not None):
        plt.xticks(range(centroids.shape[1]), xlabels, fontsize=fontsize,
                   rotation=25)
    else:
        plt.xticks(range(centroids.shape[1]),
                   range(1, centroids.shape[1]+1), fontsize=fontsize)

    if (n_clusters <= 100):
        plt.yticks(range(n_clusters), best_sol, fontsize=fontsize)
    else:
        plt.yticks(range(n_clusters), ["-"] * n_clusters)

    fig.tight_layout()
    if (filename is not None):
        plt.savefig(filename + '/' + figurename + '.png', dpi=dpi)
    fig.show()

    return best_sol


def plot_triclusters(n_clusters, arr, tri_chroms, tri_genes, tri_samples,
                     delta, l, xlabels=None, labels=None,
                     n_subplots=4, scaled=False, figsize=(17.075, 8.1125),
                     filename=None, dpi=150, paper_style=False, fontsize=8):
    """
    Plots the triclusters discovered by DeltaTrimax.

    Parameters
    ----------
    n_clusters : int
        The number of clusters.
    arr : ndarray
        The data clustered.
    tri_chroms : list of ndarray
        A list of length #triclusters, containg a boolean ndarray for each
        tricluster. The boolean array is of length #chromosomes and contains
        True if the respective chromosome is contained in the tricluster,
        False otherwise.
    tri_genes : list of ndarray
        A list of length #triclusters, containg a boolean ndarray for each
        tricluster. The boolean array is of length #genes and contains
        True if the respective gene is contained in the tricluster,
        False otherwise.
    tri_samples : list of ndarray
        A list of length #triclusters, containg a boolean ndarray for each
        tricluster. The boolean array is of length #samples and contains
        True if the respective sample is contained in the tricluster,
        False otherwise.
    delta : float
        The delta parameter of the algorithm. Must be > 0.0
    l : float
        The lambda parameter of the algorithm. Must be >= 1.0
    xlabels : list of str, default None
        The labels of the x axis. If None given, numbers are plotted.
    labels : ndarray
        The labels assigned to each datapoint during clustering.
    n_subplots : int, default 4
        The number of subplots in each window.
    scaled : bool, default False
        If True, the final plots are scaled along y-axis, ie the range of the
        y-axis in the same among all plots for better comparison.
    figsize : tuple, default (17.075, 8.1125)
        The size of the window.
    filename : str, default None
        The path in which to save the resulting plot. The plot is saved as .png
    dpi : int, default 150
        The dpi quality of the final plot (Applies only when plot is saved in
        file)
    fontsize : int, default 8
        The size of the fonts in the plot.
    """
    tri_chroms = np.vstack(tri_chroms)
    tri_genes = np.vstack(tri_genes)
    tri_samples = np.vstack(tri_samples)

    triclusters_chroms = [np.nonzero(tri_row)[0] for tri_row in tri_chroms]
    triclusters_genes = [np.nonzero(tri_row)[0] for tri_row in tri_genes]
    triclusters_samples = [np.nonzero(tri_row)[0] for tri_row in tri_samples]

    if (scaled):
        ymax = int(ceiling(np.nanmax(arr)))

    no_windows = (n_clusters // n_subplots)
    if (n_clusters % n_subplots > 0):
        no_windows += 1

    clust = 0
    for win in range(1, no_windows+1):
        figurename = "n_clusters=" + str(n_clusters) + "_d" + str(delta) + "_l" + str(l)  + "_" + str(win)
        fig = plt.figure(figurename, figsize=figsize)
        while (clust < n_clusters) and (clust < win*n_subplots):
            if ((n_clusters // n_subplots) - win < 0):
                plt.subplot(ceiling((n_clusters % n_subplots) / 2.0), 2, (clust % n_subplots)+1)
            else:
                plt.subplot(ceiling(n_subplots/2), 2, (clust % n_subplots)+1)

            tri_chrom = triclusters_chroms[clust]
            tri_gene = triclusters_genes[clust]
            tri_sample = triclusters_samples[clust]

            if (paper_style):
                sns.set_context("paper")
                sns.set_style("white")   # Make the background white

            nonnan_genes = 0

            for chrom in tri_chrom:
                for gene in tri_gene:  # for each gene in the cluster
                    plt.plot(range(len(tri_sample)),
                             arr[chrom, gene, tri_sample])
                    if (not np.isnan(arr[chrom, gene, :]).all()):
                        nonnan_genes += 1
            plt.title("Cluster " + str(clust) + ": " + str(nonnan_genes) + " genes from " + str(tri_chrom.shape[0]) + " chromosome pairs",
                      fontname='Times New Roman', fontsize=fontsize+2)

            plt.xlim(0, len(tri_sample)-1)
            if (xlabels is not None):
                plt.xticks(range(len(tri_sample)),
                           xlabels[tri_sample],
                           rotation=25)
            plt.xticks(fontsize=fontsize, fontname='Times New Roman')
            if (scaled):
                plt.yticks(range(ymax), fontsize=fontsize, fontname='Times New Roman')
            else:
                plt.yticks(fontsize=fontsize, fontname='Times New Roman')

            clust += 1
            if (paper_style):
                sns.despine(fig)

        fig.tight_layout()
        if (filename is not None):
            plt.savefig(filename + '/' + figurename + '.png', dpi=dpi)
        fig.show()


def plot_tricentroids(n_clusters, arr, tri_chroms, tri_genes, tri_samples,
                      delta, l, xlabels=None, n_subplots=4, error_bars='range',
                      figsize=(17.075, 8.1125), filename=None, dpi=150,
                      paper_style=False, fontsize=8):
    """
    Plots the centroids of the triclusters discovered by DeltaTrimax.

    Parameters
    ----------
    n_clusters : int
        The number of clusters.
    arr : ndarray
        The data clustered.
    tri_chroms : list of ndarray
        A list of length #triclusters, containg a boolean ndarray for each
        tricluster. The boolean array is of length #chromosomes and contains
        True if the respective chromosome is contained in the tricluster,
        False otherwise.
    tri_genes : list of ndarray
        A list of length #triclusters, containg a boolean ndarray for each
        tricluster. The boolean array is of length #genes and contains
        True if the respective gene is contained in the tricluster,
        False otherwise.
    tri_samples : list of ndarray
        A list of length #triclusters, containg a boolean ndarray for each
        tricluster. The boolean array is of length #samples and contains
        True if the respective sample is contained in the tricluster,
        False otherwise.
    delta : float
        The delta parameter of the algorithm. Must be > 0.0
    l : float
        The lambda parameter of the algorithm. Must be >= 1.0
    xlabels : list of str, default None
        The labels of the x axis. If None given, numbers are plotted.
    n_subplots : int, default 4
        The number of subplots in each window.
    error_bars : {'sd', 'error'}, default 'error'
        The type of the error bars in the plot.
    figsize : tuple, default (17.075, 8.1125)
        The size of the window.
    filename : str, default None
        The path in which to save the resulting plot. The plot is saved as .png
    dpi : int, default 150
        The dpi quality of the final plot (Applies only when plot is saved in
        file)
    fontsize : int, default 8
        The size of the fonts in the plot.
    """
    tri_chroms = np.vstack(tri_chroms)
    tri_genes = np.vstack(tri_genes)
    tri_samples = np.vstack(tri_samples)

    triclusters_chroms = [np.nonzero(tri_row)[0] for tri_row in tri_chroms]
    triclusters_genes = [np.nonzero(tri_row)[0] for tri_row in tri_genes]
    triclusters_samples = [np.nonzero(tri_row)[0] for tri_row in tri_samples]

    ymax = int(ceiling(np.nanmax(arr)))

    no_windows = (n_clusters // n_subplots)
    if (n_clusters % n_subplots > 0):
        no_windows += 1

    clust = 0
    with warnings.catch_warnings():  # We expect mean of NaNs here
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for win in range(1, no_windows+1):
            figurename = "Centroids of n_clusters=" + str(n_clusters) + "_d" + str(delta) + "_l" + str(l) + "_" + str(win) + '_' + error_bars
            fig = plt.figure(figurename, figsize=figsize)
            while (clust < n_clusters) and (clust < win*n_subplots):
                if ((n_clusters // n_subplots) - win < 0):
                    plt.subplot(ceiling((n_clusters % n_subplots) / 2.0), 2, (clust % n_subplots)+1)
                else:
                    plt.subplot(ceiling(n_subplots/2), 2, (clust % n_subplots)+1)

                plt.title('Cluster ' + str(clust), fontname='Times New Roman', fontsize=fontsize+2)

                if (paper_style):
                    sns.set_context("paper")
                    sns.set_style("white")   # Make the background white

                tri_chrom = triclusters_chroms[clust]
                tri_gene = triclusters_genes[clust]
                tri_sample = triclusters_samples[clust]      

                if (xlabels is not None):
                    plt.xticks(range(len(tri_sample)),
                               xlabels[tri_sample],
                               rotation=25)
                plt.xticks(fontsize=fontsize, fontname='Times New Roman')
                plt.xlim(0, len(tri_sample)-1)

                # Get the slice of the 3D array corresponding to the
                # tricluster
                tricluster = arr[tri_chrom, :, :]
                tricluster = tricluster[:, tri_gene, :]
                tricluster = tricluster[:, :, tri_sample]

                # Compute the centroid and the errors for each sample
                # and plot the results
                centroid = np.nanmean(np.nanmean(tricluster, axis=1), axis=0)
                if (error_bars == 'range'):
                    divergence = tricluster - np.broadcast_to(centroid, (len(tri_chrom), len(tri_gene), len(tri_sample)))
                    errors = []
                    errors.append(np.absolute(np.nanmin(np.nanmin(divergence, axis=1), axis=0)))
                    errors.append(np.absolute(np.nanmax(np.nanmax(divergence, axis=1), axis=0)))
                else:
                    errors = np.nanstd(np.nanstd(tricluster, axis=1), axis=0)
                plt.errorbar(range(len(tri_sample)), centroid, yerr=errors)

                plt.yticks(range(ymax), fontsize=fontsize, fontname='Times New Roman')

                clust += 1
                if (paper_style):
                    sns.despine(fig)

            fig.tight_layout()
            if (filename is not None):
                plt.savefig(filename + '/' + figurename + '.png', dpi=dpi)
            fig.show()
