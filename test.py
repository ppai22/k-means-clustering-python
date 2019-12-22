import pandas as pd
from Clustering import KMeans


def load_iris():
    """Method to load the iris data set from CSV file"""
    # Path to CSV file
    file_path = r'iris.csv'
    # Load data using pandas
    data = pd.read_csv(file_path)
    # Dropping the classification parameter
    data_points = data.drop(['variety'], axis=1)
    # Return the required data in the form of a list as per code requirements in other methods
    return data_points.values.tolist()


if __name__ == '__main__':
    # Load Iris dataset
    data_set = load_iris()
    # Instantiate object for clustering
    model = KMeans(data_set, 3)
    # Run the clustering algorithm
    model.fit()
    # Print the cluster points
    print(model.clusters)
    # Print sizes of each cluster
    print(model.cluster_sizes)
