# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def read_data_from_excel(filename):

    # Read data from given excel spreadsheet
    data = pd.read_excel(filename)

    # Create a proper DataFrame structure
    # Row indexes = (row number, ENTREZ_GENE_ID, SYMBOL)
    # Column labels = the sample labels

    # Use row 0 as labels
    newlbl = data.ix[0].copy()
    newlbl[0] = 'ENTREZ_GENE_ID'
    newlbl[1] = 'SYMBOLS'
    newlbl[2] = 'Chromosome'
    data.columns = newlbl.values

    # Delete rows 0 and 1
    data = data.ix[2:]

    # Replace strings with integers in Chromosome names
    # X => 23
    # Y => 23
    data['Chromosome'] = data['Chromosome'].map({'1':1, '2':2, '3':3, '4':4, 
                                                 '5':5, '6':6, '7':7, '8':8, 
                                                 '9':9, '10':10, '11':11, 
                                                 '12':12, '13':13, '14':14, 
                                                 '15':15, '16':16, '17':17, 
                                                 '18':18, '19':19, '20':20, 
                                                 '21':21, '22':22, 'X':23, 
                                                 'Y':23})

    # Fix symbols
    new_symbols = []
    for symbol in data['SYMBOLS'].values:
        new_symbols.append(symbol.partition('\\')[0])
    data['SYMBOLS'] = new_symbols

    # Drop rows with NaNs in chromosomes (i.e. those unmapped)
    data.dropna(subset=['Chromosome'], inplace=True)

    # Delete NaN rows
    data.dropna(how='all', subset=data.columns[3:], inplace=True)

    # Set row indexes as ENTREZ_GENE_ID and SYMBOLS
    newind1 = data['ENTREZ_GENE_ID'].values.copy()
    newind2 = data['SYMBOLS'].values.copy()

    data.index = [np.arange(len(data.values)), newind1, newind2]
    data.index.names = ['index', 'ENTREZ_GENE_ID', 'SYMBOL']
    data.drop('ENTREZ_GENE_ID', axis=1, inplace=True)
    data.drop('SYMBOLS', axis=1, inplace=True)

    # Convert dtype of values into float
    # (for better handling by the clustering algorithms)
    data[:] = data[:].astype(float)

    return data


def save_cluster_genes(filename, gene_ids, algo='k-means', mode=0, 
                       clusters=None, tri_chroms=None, tri_genes=None,
                       tri_samples=None, sample_labels=None):
    '''
    Saves the id of the genes in all clusters into a certain .txt file.
    
    Output format:
    "
    Cluster CLUSTER_ID:
    GENE_ID_1
    GENE_ID_2,
    ...
    "
    '''

    f = open(filename, 'a')

    if (algo == 'triclustering'):
        triclusters_chroms = [np.nonzero(tri_row)[0] for tri_row in tri_chroms]
        triclusters_genes = [np.nonzero(tri_row)[0] for tri_row in tri_genes]
        triclusters_samples = [np.nonzero(tri_row)[0] for tri_row in tri_samples]

        for clust in range(len(triclusters_chroms)):
            f.write('Cluster ' + str(clust) + ':\n')

            for chrom in triclusters_chroms[clust]:
                for gene in triclusters_genes[clust]:
                    if (gene_ids[chrom][gene] is not None):
                        f.write(str(gene_ids[chrom][gene][2]) + '\n')

            f.write('\nSAMPLES:\n')
            for sample in triclusters_samples[clust]:
                f.write(sample_labels[sample] + '\n')

            f.write('\n---\n')
    elif (algo == 'k-means'):
        if (mode == 0) or (mode == 1):
            for clust in clusters.keys():
                f.write('Cluster ' + str(clust) + ':\n')

                for gene_group in clusters[clust]:
                    for chrom in range(len(gene_ids)):
                        gene = gene_ids[chrom][gene_group]
                        if (gene is not None):
                            f.write(str(gene[2]) + '\n')

                f.write('\n---\n')
        else:
            for clust in clusters.keys():
                f.write('Cluster ' + str(clust) + ':\n')

                for sample in clusters[clust]:
                    f.write(sample_labels[sample] + '\n')

                f.write('\n---\n')

    else:    # algo == 'array3D'
        n_chroms = len(gene_ids)
        for sl in range(len(gene_ids[0])):
            # For each slice
            f.write('Slice ' + str(sl) + ':\n')

            for chrom in range(n_chroms):
                # For each gene in slice
                gene = gene_ids[chrom][sl]
                if (gene is not None):
                    f.write(str(gene[2]) + '\n')

            f.write('\n---\n')

    f.close()
