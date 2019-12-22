# K-Means Clustering implemented using Python
Test script that performs K-Means clustering of given data set using Python without using frameworks

## Dependencies:
- Python modules: numpy, random, pandas, time
- Python version: 3.7.4

Inputs required:
- data_set: Input data set in given format as demonstrated in the code
- k_value: No of clusters needed
- precision: The precision with which calculations have to be made

## Files
```KMeansClustering.py``` is a single script which runs the clustering<br>
```Clustering.py``` is the same script written as a class and can be called with ```from Clustering import KMeans```<br>
```test.py``` is an implementation of the algorithm using the class on the [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)

## Further improvements:
- Code is slow for huge data sets. Need to optimize.
- Need to compare with same data set run on frame work such scikitlearn and compare the results
