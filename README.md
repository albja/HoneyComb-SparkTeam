# SparkTeam
basic version (NOT completed)

1.Start spark via the command line with parameters, the input
parameters are two local files path.

2.Use spark to train a machine learning model, and make predictions.

3.Save result path as JSON

INSTRUCTION:
1. Download Spark from :http://spark.apache.org/downloads.html (version:1.5.2)
   please choose the approriate package type according to Hadoop version.

2. To build Spark and its example programs, run:
   build/mvn

3. Install python 2.7(should also support Python 3)

3. Put Kmeans.py and kmeans_data.txt to some directory and run command line:
   YOUR_SPARK_PATH/spark-submit --master local[4] K_means.py
   YOUR_INPUT_PATH/kmeans_data.txt