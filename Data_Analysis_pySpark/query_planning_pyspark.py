import os
os.chdir("/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/chapter 11")
from pyspark.sql import SparkSession
import pyspark.sql.functions as F 

# access SPARK UI at localhost:4040
# SPARK UI explaination: p 246 - p 251

# create a data frame
spark = SparkSession.builder.appName(
    "Counting words from a book").getOrCreate()

base_path = "/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/data" \
    "/DataAnalysisWithPythonAndPySpark-Data-trunk"
book_path = os.path.join(base_path,
                         "gutenberg_books",
                         "*.txt")

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

results.orderBy(F.col("count").desc()).show(10)

#%% relaunching PySpark to change # cores and # RAM available
spark = (
    SparkSession.builder.appName("Launching pyspark with cunstom options")
    .master("local[8]") # using 8 cores for the master
    .config("spark.driver.memory", "2g") # the driver will use 2 gb
    )

#%%