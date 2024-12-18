import os
os.chdir("/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/chapter 7")
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException # exceptions
import pyspark.sql.functions as F
import pyspark.sql.types as T

# read the data

spark = SparkSession.builder.getOrCreate()

base_dir = "/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/data/" \
    "DataAnalysisWithPythonAndPySpark-Data-trunk/elements"


elements = spark.read.csv(
    os.path.join(base_dir, "Periodic_Table_Of_Elements.csv"),
    header=True,
    inferSchema=True,
)

# reading and counting the liquid elements by period
""" pyspark """
elements.filter(
    F.col("phase") == 'liq').groupBy(F.col("period")).count().show()

""" SQL """
"""
SELECT
    period, COUNT(*)
FROM elements
WHERE phase = 'liq'
GROUP BY period;
"""

""" promoting spark dataframes to spark tables """
# failing to query with SQL
try:
    spark.sql(
        "SELECT period, COUNT(*) FROM elements "
        "WHERE phase = 'liq' GROUP BY period").show(5)
except AnalysisException as a:
    print(a)
    
# succeeding to query with SQL
elements.createOrReplaceTempView("elements") # register as SQL table

spark.sql(
    "SELECT period, COUNT(*) AS count FROM elements WHERE phase = 'liq' GROUP BY period").show(5)


# show and drop registered tables
print(spark.catalog.listTables())

spark.catalog.dropTempView("elements")

print(spark.catalog.listTables())