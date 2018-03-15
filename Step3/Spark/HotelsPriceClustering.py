import sys
import argparse
from pyspark.sql import SparkSession
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.feature import VectorAssembler
from scipy.cluster import hierarchy as hc
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist


def view_trim(text, length):
    if len(text) > length:
        return text[:length - 3] + '...'
    return text


def create_kmeans_dendogram(input_csv, num_clusters):
    spark = SparkSession.builder.appName('HotelsPriceDataGeneratorSession').getOrCreate()

    # Lazy op - Load the data
    # Read CSV
    # Note that the schema is already defined, a fully null df will result if the csv does not fit the schema
    print('Reading CSV from ' + input_csv)
    generated_hotels_df = spark.read.csv(
        input_csv, header=True,
        inferSchema=True)

    # Limit the clusters to num cols
    num_clusters = min(num_clusters, len(generated_hotels_df.columns[1:]))

    # Assemble the features vector column
    vecAssembler = VectorAssembler(inputCols=generated_hotels_df.columns[1:], outputCol="features")
    vector_df = vecAssembler.transform(generated_hotels_df)

    # Run the BisectingKMeans to find hierarchial clusters
    kmeans = BisectingKMeans().setK(num_clusters).setSeed(42)
    model = kmeans.fit(vector_df)

    # Link it to find relations between the clusters
    z = hc.linkage(model.clusterCenters(), method='average', metric='correlation')

    # Plot the dendrogram
    hc.dendrogram(z)
    plt.show()


def create_pdist_dendogram(input_csv):
    spark = SparkSession.builder.appName('HotelsPriceDataGeneratorSession').getOrCreate()

    # Lazy op - Load the data
    # Read CSV
    # Note that the schema is already defined, a fully null df will result if the csv does not fit the schema
    print('Reading CSV from ' + input_csv)
    generated_hotels_df = spark.read.csv(
        input_csv, header=True,
        inferSchema=True)

    # Assemble the features vector column
    vecAssembler = VectorAssembler(inputCols=generated_hotels_df.columns[1:], outputCol="features")
    vector_df = vecAssembler.transform(generated_hotels_df)

    # Create distance vector from the dataframe features
    distvec = pdist(np.array(vector_df.select(generated_hotels_df.columns[1:]).collect()))

    # Find the hierarchy linkage in an average method correlated
    z = hc.linkage(distvec, method='average', metric='correlation')

    # Plot the dendrogram
    hc.dendrogram(z,
                  labels=np.array([v['Hotel Name']
                                   for v in
                                   generated_hotels_df.select('Hotel Name').collect()]))
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
