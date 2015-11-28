import sys
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SQLContext
from pyspark.sql import DataFrame
from numpy import array
from math import sqrt

#"/Users/jacobliu/HoneyPySpark/spark-1.5.2-bin-hadoop2.6/README.md"  # Should be some file on your system
sc = SparkContext("local", "KMeans")

loadFilePath = sys.argv[1]		#input file path
dumpFilePath = sys.argv[2]		#output file path

# Load and parse the data
data = sc.textFile(loadFilePath)
parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))

# Build the model (cluster the data)
clusters = KMeans.train(parsedData, 3, maxIterations=10, runs=30, initializationMode="random")

# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

#save output file path as JSON and dump into dumpFilePath
SQLContext = SQLContext(sc)
res = [('k_means',dumpFilePath)]
rdd = sc.parallelize(res)
SQLContext.createDataFrame(rdd).collect()
df = SQLContext.createDataFrame(rdd,['model_name','res_path'])
df.toJSON().saveAsTextFile(dumpFilePath)


# Save and load model
#clusters.save(sc, "myModel")
#sameModel = KMeansModel.load(sc, "myModel")