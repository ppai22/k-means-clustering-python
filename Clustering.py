import numpy as np
import random


class KMeans:

    def __init__(self, data, k, precision=5):
        self.data = data
        self.k = k
        self.precision = precision
        self.centroids = []
        self.clusters = []
        self.cluster_sizes = []

    @staticmethod
    def dist(p1, p2):
        """Method returns the euclidean distance between two points"""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    @staticmethod
    def cal_centroids(_, precision):
        """Method returns centroids for input list of clusters"""
        centroids = []
        for item in _:
            ele = len(item[0])
            c_item = []
            for i in range(ele):
                c_item.append(round(np.mean([li[i] for li in item]), precision))
            centroids.append(c_item)
        return centroids

    @staticmethod
    def compare_lists(list_1, list_2):
        """Method returns True when the two lists are same else returns False"""
        flag = True
        for item in list_1:
            if item not in list_2:
                flag = False
            break
        if flag:
            return True
        else:
            return False

    def fit(self):
        """Method that does the clustering"""
        # Randomly selects indices of initial centroids
        indices = random.sample(range(len(self.data)), self.k)
        # Initialization of initial centroids
        for index in indices:
            self.centroids.append(self.data[index])
        # Loop that does the iterative calculations
        while True:
            # Initialization of clusters
            self.clusters = []
            for i in range(self.k):
                self.clusters.append([])
            # Calculation of Euclidean distances between each point in the data set and the centroids
            for point in self.data:
                distances = []
                for centroid in self.centroids:
                    distances.append(self.dist(point, centroid))
                # Clustering of each point happens in this step
                self.clusters[distances.index(min(distances))].append(point)
            # Calculation of new centroids
            new_centroids = self.cal_centroids(self.clusters, self.precision)
            # Loop breaks when centroids do not change any more
            if self.compare_lists(self.centroids, new_centroids):
                break
            # Updating new centroids for next iteration
            self.centroids = new_centroids
        # Cluster sizes
        for cluster in self.clusters:
            self.cluster_sizes.append(len(cluster))
