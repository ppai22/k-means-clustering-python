import numpy as np
import random
import time


def calc_time_taken(method):
    """Decorator method to calculate time taken inside a method"""
    def wrapper(*args):
        # Start clock
        start = time.time()
        # Run the method
        method(*args)
        # Stop clock
        end = time.time()
        # Time taken
        print("Time taken: " + str(end-start) + " s")
    return wrapper


def dist(p1, p2):
    """Method returns the euclidean distance between two points"""
    return np.linalg.norm(np.array(p1)-np.array(p2))


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


@calc_time_taken
def runner(data, k, prec=5):
    """Runner method for K-Means Clustering"""
    # Randomly selects indices of initial centroids
    indices = random.sample(range(len(data)), k)
    # Initialization of initial centroids
    centroids = []
    for index in indices:
        centroids.append(data[index])
    # Initialization of counter value to count no of iterations needed
    counter = 0
    # Loop that does the iterative calculations
    while True:
        # Updates the counter value
        counter += 1
        # Initialization of clusters
        clusters = []
        for i in range(k):
            clusters.append([])
        # Calculation of Euclidean distances between each point in the data set and the centroids
        for point in data:
            distances = []
            for centroid in centroids:
                distances.append(dist(point, centroid))
            # Clustering of each point happens in this step
            clusters[distances.index(min(distances))].append(point)
        # Calculation of new centroids
        new_centroids = cal_centroids(clusters, prec)
        # Printing of old and new centroids for debugging purpose only (Can be removed)
        print(centroids)
        print(new_centroids)
        # Loop breaks when centroids do not change any more
        if compare_lists(centroids, new_centroids):
            break
        # Updating new centroids for next iteration
        centroids = new_centroids
        # Printing counter value for debugging purpose only
        print(counter)
    # Prints the Centroids of the final clusters
    print(centroids)
    # Prints values in each cluster and no of items in each cluster
    for cluster in clusters:
        print(cluster)
        print(len(cluster))


if __name__ == '__main__':
    # Input data set needs to be in following format
    data_set = [[2, 3], [3, 5], [5, 8], [-3, 0], [9, 0], [5, 1], [4, -2], [4, 0], [9, -3], [4, 7], [8, 6], [9, 4],
                [-2, 6], [8, 2], [2, -2], [-2, 2]]
    # Value of K
    k_value = 2
    # Precision used for calculations
    precision = 5
    # Function call to call the runner method
    runner(data_set, k_value, precision)
