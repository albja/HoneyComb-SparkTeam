# SparkTeam
Version 1.0

1.Start spark via the command line with parameters, the input
parameters are intput/output files path.

2.Use spark to train a machine learning model, and make predictions.

3.Save result as JSON into output path

4.Support different machine learning models(KMeans, Logistic Regression).

INSTRUCTION:

1. Download Spark from :http://spark.apache.org/downloads.html (version:1.5.2)
   please choose the approriate package type according to Hadoop version.

2. To build Spark and its example programs, run:
   build/mvn

3. Install python 2.7(should also support Python 3)

4. Login Cluster: 
   ssh honeycomb@128.2.7.38 (password: ask teammates)

5. Copy files into CLuster local host: 
   scp source_file_name honeycomb@128.2.7.38:/home/honeycomb/SparkTeam

6. Put files into HDFS:
   hdfs dfs -put LOCAL_FILE_PATH HDFS_FILE_PATH
   
   e.g:
   
   hdfs dfs -put /home/honeycomb/SparkTeam/sample_multiclass_classification_data_test.txt /user/honeycomb/sparkteam/input

7. Put PySpark.py and train/test dataset into HDFS and run command line:
   YOUR_SPARK_PATH/spark-submit PySpark.py YOUR_TRAIN_DATA_PATH YOUT_TEST_DATA_PATH YOUR_OUTPUT_PATH
   
   e.g:
   
   /bin/spark-submit /home/honeycomb/SparkTeam/PySpark.py /user/honeycomb/sparkteam/input/sample_multiclass_classification_data.txt /user/honeycomb/sparkteam/output1 Regression

Resource:

1. Deploy Spark: http://spark.apache.org/docs/latest/programming-guide.html

2. Hadoop Version: http://spark.apache.org/docs/latest/hadoop-third-party-distributions.html

3. Python API Docs: https://spark.apache.org/docs/1.5.2/api/python/index.html

4. Machine Learning Library(MLib) Guide: http://spark.apache.org/docs/latest/mllib-guide.html