import os 
os.chdir("/Users/pepijnschouten/Desktop/Deep Learning Projects/Pyspark/chapter 2")

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

base_path = "/Users/pepijnschouten/Desktop/Deep Learning Projects/Pyspark/"
book_path = os.path.join(base_path,
                         "data",
                         "DataAnalysisWithPythonAndPySpark-Data-trunk",
                         "gutenberg_books",
                         "1342-0.txt")

""" create spark session """
spark = SparkSession.builder.getOrCreate()

""" create eager spark session """
# spark = SparkSession  \
#     .builder \
#         .config("spark.sql.repl.eagerEval.enabled", "True") \
#             .getOrCreate()

print('spark read object', spark.read)
print(dir(spark.read))

""" read the text file to data frame """
book = spark.read.text(book_path)

print(book)

""" view dtypes """
dtypes = book.dtypes

""" view schema """
schema = book.printSchema()

print('dtypes: ', dtypes, 'schema: ', schema)

""" view part of the dataframe """
book.show(
    n=20,
    truncate=50,
    vertical=False)

""" four ways of selecting columns """
book.select(book.value)
book.select(book["value"])
book.select(F.col("value")) # prefered
book.select("value")

""" renaming columns """
book = book.withColumnRenamed("value", "renamed")
book.show()
book = book.withColumnRenamed("renamed", "value")

""" selecting columns, splitting on spaces, rename transformed column """
lines = book.select(F.split(book.value, ' ').alias("line"))

lines.show(n=5)
print(lines.printSchema())

""" one word per row """
words = lines.select(F.explode(F.col("line")).alias("word"))

words.show(15)

""" lower case words """
words_lower = words.select(F.lower(F.col("word")).alias('word_lower'))

words_lower.show(15)

""" extract only words with regex """
words_clean = words_lower.select(
    F.regexp_extract(F.col("word_lower"), "[a-z]+", 0).alias("word"))

words_clean.show()

""" remove empty strings """
words_nonull = words_clean.filter(F.col("word") != "")

words_nonull.show()

""" grouping the data frame """
groups = words_nonull.groupBy(F.col("word"))

print(groups)


""" create word counter """
results = groups.count()

results.show()

""" order the words by a column """
results.orderBy("count", ascending=False).show(10)
results.orderBy(F.col("count").desc()).show(10)

""" write results to csv """
write_path = os.path.join(base_path,
                          "exports",
                          "simple_count.csv")
results.write.csv(write_path)


""" reduce partitions """
write_path_2 = os.path.join(base_path,
                          "exports",
                          "simple_count_single_partition.csv")
results.coalesce(1).write.csv(write_path_2)

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
