from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import os

spark = SparkSession.builder.getOrCreate()

my_list = [
    ["Banana", 2, 1.74],
    ["Apple", 4, 2.04],
    ["Carrot", 1, 1.09],
    ["Cake", 1, 10.99],
    ]

""" create dataframe """
df = spark.createDataFrame(
    my_list, ["Item", "Qunatity", "Price"])

df.printSchema()


""" reading a csv file with infered schema """
base_dir = "/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/" \
           "data/DataAnalysisWithPythonAndPySpark-Data-trunk/broadcast_logs"

logs = spark.read.csv(
    os.path.join(base_dir,"BroadcastLogs_2018_Q3_M8_sample.CSV"),
    sep="|",
    header=True,
    inferSchema=True,
    timestampFormat="yyyy-MM-dd")

logs.printSchema()

""" selecting columns """
logs_selection = logs.select(*["BroadCastLogID", "LogDate"])

logs_selection.show(5, truncate=False)

""" deleting columns """
logs = logs.drop(F.col("BroadcastLogID"), F.col("SequenceNO"))

print("BroadcastLogID" in logs.columns)
print("SequenceNO" in logs.columns)

""" creating new columns """
logs.select(
    F.col("Duration"),
    F.col("Duration").substr(1, 2).cast("int").alias("dur_hours"),
    F.col("Duration").substr(4, 2).cast("int").alias("dur_minutes"),
    F.col("Duration").substr(7, 2).cast("int").alias("dur_seconds"),
    ).distinct().show(5)

logs = logs.withColumn(
    "Duration_seconds",
    (
         F.col("Duration").substr(1, 2).cast("int") * 60 * 60
         + F.col("Duration").substr(4, 2).cast("int") * 60
         + F.col("Duration").substr(7, 2).cast("int")
     ),
)

logs.printSchema()

""" renaming columns """
logs = logs.withColumnRenamed("Duration_seconds", "duration_seconds")

logs.printSchema()

""" renaming all columns """
logs.toDF(*[x.lower() for x in logs.columns]).printSchema()

""" reordering columns """
logs.select(sorted(logs.columns)).printSchema()

""" diagnising data frame """
for i in logs.columns:
    logs.describe(i).show()
    
for i in logs.columns:
    logs.select(i).summary("min", "max").show()
