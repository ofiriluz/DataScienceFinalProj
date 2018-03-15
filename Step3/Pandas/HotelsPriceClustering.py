import sys
import argparse
import pandas as pd
from scipy.cluster import hierarchy as hc
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans


def view_trim(text, length):
    if len(text) > length:
        return text[:length - 3] + '...'
    return text


def create_kmeans_dendogram(input_csv, num_clusters):
    hotels_df = pd.read_csv(input_csv)

    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(hotels_df[hotels_df.columns[1:]].as_matrix())

    # Link it to find relations between the clusters
    z = hc.linkage(kmeans.cluster_centers_, method='average', metric='correlation')

    # Plot the dendrogram
    hc.dendrogram(z)
    plt.show()


def create_pdist_dendogram(input_csv):
    hotels_df = pd.read_csv(input_csv)

    # Create distance vector from the dataframe features
    distvec = pdist(hotels_df[hotels_df.columns[1:]].as_matrix())

    # Find the hierarchy linkage in an average method correlated
    z = hc.linkage(distvec, method='average', metric='correlation')

    # Plot the dendrogram
    hc.dendrogram(z,
                  labels=np.array(hotels_df['Hotel Name'].tolist()))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hotels Price Hierarchical Clustering')
    parser.add_argument('--input_csv', required=True, help='Input CSV of hotel prices to cluster on', type=str)
    subparsers = parser.add_subparsers(dest='mode', description='Clustering mode')
    kmeans_parser = subparsers.add_parser('KMeans')
    pdist_parser = subparsers.add_parser('PDist')

    kmeans_parser.add_argument('--num_clusters', default=6, help='Amount of clusters for the bisecting kmeans',
                               type=int)

    args = parser.parse_args(sys.argv[1:])

    if args.mode == 'KMeans':
        create_kmeans_dendogram(args.input_csv, args.num_clusters)
    else:
        create_pdist_dendogram(args.input_csv)
