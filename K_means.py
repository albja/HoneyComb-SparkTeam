import sys
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans
from numpy import array
from math import sqrt
#"/Users/jacobliu/HoneyPySpark/spark-1.5.2-bin-hadoop2.6/README.md"  # Should be some file on your system
sc = SparkContext("local", "KMeans")

logFile = sys.argv[1]

# Load and parse the data
data = sc.textFile(logFile)
parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))

# Build the model (cluster the data)
clusters = KMeans.train(parsedData, 2, maxIterations=10, runs=30, initializationMode="random")

# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

print("clusters:")
print clusters.labels_
# Save and load model
clusters.save(sc, "myModel")
#sameModel = KMeansModel.load(sc, "myModel")