import os 
os.chdir("/Users/pepijnschouten/Desktop/Deep Learning Projects/Pyspark/chapter 2")

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

base_path = "/Users/pepijnschouten/Desktop/Deep Learning Projects/Pyspark/"
book_path = os.path.join(base_path,
                         "data",
                         "DataAnalysisWithPythonAndPySpark-Data-trunk",
                         "gutenberg_books",
                         "*.txt")

""" create spark session """
spark = SparkSession.builder.getOrCreate()

""" using method chaining """
results = (
    spark.read.text(book_path)
    .select(F.split(F.col("value"), " ").alias("line"))
    .select(F.explode(F.col("line")).alias("word"))
    .select(F.lower(F.col("word")).alias("word"))
    .select(F.regexp_extract(F.col("word"), "[a-z']*", 0).alias("word"))
    .filter(F.col("word") != "")
    .groupby("word")
    .count()
)

""" reduce partitions """
write_path_2 = os.path.join(base_path,
                          "exports",
                          "scaled_simple_count_single_partition.csv")
results.coalesce(1).write.csv(write_path_2)


