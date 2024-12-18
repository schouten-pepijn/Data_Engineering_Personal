import os
os.chdir("/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/chapter 7")
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException # exceptions
import pyspark.sql.functions as F
import pyspark.sql.types as T
from functools import reduce

#%% read the data
spark = SparkSession.builder.getOrCreate()

data_dir = "/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/data" \
    "/DataAnalysisWithPythonAndPySpark-Data-trunk/backblaze"
    
q1 = spark.read.csv(
    os.path.join(data_dir, "data_Q1_2019"),
    header=True, inferSchema=True
)
q2 = spark.read.csv(
    os.path.join(data_dir, "data_Q2_2019"),
    header=True, inferSchema=True
)
q3 = spark.read.csv(
    os.path.join(data_dir, "data_Q3_2019"),
    header=True, inferSchema=True
)
q4 = spark.read.csv(
    os.path.join(data_dir, "data_Q4_2019"),
    header=True, inferSchema=True
)

# remove redundant fields
q4_fields_extra = set(q4.columns) - set(q1.columns)

for i in q4_fields_extra:
    q1 = q1.withColumn(i, F.lit(None).cast(T.StringType()))
    q2 = q2.withColumn(i, F.lit(None).cast(T.StringType()))
    q3 = q3.withColumn(i, F.lit(None).cast(T.StringType()))
    
# combine quarter data
backblaze_2019 = (
    q1.select(q4.columns)
        .union(q2.select(q4.columns))
        .union(q3.select(q4.columns))
        .union(q4)
    )

# setting layout accorindg to the schema
backblaze_2019 = backblaze_2019.select(
    [
         F.col(x).cast(T.LongType()) if x.startswith("smart") else F.col(x)
         for x in backblaze_2019.columns
     ]
)

#%%  Comparing select and where in SQL and pyspark
backblaze_2019.createOrReplaceTempView("backblaze_2019")

# SQL
spark.sql(
    "SELECT serial_number FROM backblaze_2019 WHERE failure = 1"
    ).show(5)

# pyspark
backblaze_2019.where("failure = 1").select(F.col("serial_number")).show(5)

#%%  Comparing grouping and ordering in SQL and pyspark

# SQL
spark.sql(
    """SELECT
            model,
            MIN(capacity_bytes / POW(1024, 3)) AS min_GB,
            MAX(capacity_bytes / POW(1024, 3)) AS max_GB
        FROM
            backblaze_2019
        GROUP BY
            model
        ORDER BY
            max_GB DESC
    """
).show(5)

# pyspark
backblaze_2019.groupBy(F.col("model")).agg(
    F.min(F.col("capacity_bytes") / F.pow(F.lit(1024), 3)).alias("min_GB"),
    F.max(F.col("capacity_bytes") / F.pow(F.lit(1024), 3)).alias("max_GB")
    ).orderBy(F.col("max_GB"), ascending=False).show(5)

#%% USING HAVING IN SQL AND WHERE IN PYSPARK

# SQL
spark.sql(
    """SELECT
            model,
            MIN(capacity_bytes / POW(1024, 3)) AS min_GB,
            MAX(capacity_bytes/ POW(1024, 3)) AS max_GB
        FROM 
            backblaze_2019
        GROUP BY 
            model
        HAVING
            min_GB != max_GB
        ORDER BY
            max_GB DESC
    """
).show(5)

# pyspark
backblaze_2019.groupBy(F.col("model")).agg(
    F.min(F.col("capacity_bytes") / F.pow(F.lit(1024), 3)).alias("min_GB"),
    F.max(F.col("capacity_bytes") / F.pow(F.lit(1024), 3)).alias("max_GB")
).where(F.col("min_GB") != F.col("max_GB")).orderBy(
    F.col("max_GB"), ascending=False
).show(5)

    
#%% CREATING NEW TABLES/VIEWS
backblaze_2019.createOrReplaceTempView("drive_stats")

# SQL
drive_days = spark.sql(
    """
    CREATE or REPLACE TEMPORARY VIEW drive_days AS
        SELECT model, COUNT(*) AS drive_days
        FROM drive_stats
        GROUP BY model
    """
    )

failures = spark.sql(
    """
    CREATE or REPLACE TEMPORARY VIEW failures AS
        SELECT model, COUNT(*) as failures
        FROM drive_stats
        WHERE failure = 1
        GROUP BY model
    """
        )

# pyspark
drive_days = backblaze_2019.groupBy(F.col("model")).agg(
    F.count(F.col("*")).alias("drive_stats"))

failures = (backblaze_2019.where(F.col("failure") == 1)
            .groupBy(F.col("model"))
            .agg(F.count("*")).alias("failures"))

#%% UNIONING TABLES IN SQL AND PYSPARK
columns_backblaze = ", ".join(q4.columns)

q1.createOrReplaceTempView("Q1")
q2.createOrReplaceTempView("Q2")
q3.createOrReplaceTempView("Q3")
q4.createOrReplaceTempView("Q4")

# SQL
spark.sql(
    """
    CREATE OR REPLACE TEMPORARY VIEW backblaze_2019 AS
    SELECT {col} FROM Q1 UNION ALL
    SELECT {col} FROM Q2 UNION ALL
    SELECT {col} FROM Q3 UNION ALL
    SELECT {col} FROM Q4
    """.format(col=columns_backblaze)
)

# pyspark
backblaze_2019 = (
    q1.select(q4.columns)
        .union(q2.select(q4.columns))
        .union(q3.select(q4.columns))
        .union(q4)
)


#%% JOINING TABLES IN SQL AND PYSPARK

# SQL
spark.sql(
    """
        SELECT
            drive_days.model,
            drive_days,
            failures
        FROM
            drive_days
        LEFT JOIN failures
        ON
            drive_days.model = failures.model
    """
).show(5)

# pyspark
drive_days.join(failures, on="model", how="left").show(5)


#%% USING SQL SUBQUERIES

spark.sql(
    """
        SELECT
            failures.model,
            failures / drive_days AS failure_rate
        FROM (
            SELECT
                model,
                COUNT(*) as drive_days
            FROM drive_stats
            GROUP BY model) AS drive_days
        INNER JOIN (
            SELECT
                model,
                COUNT(*) AS failures
            FROM drive_stats
            WHERE failure = 1
            GROUP BY model) AS failures
        ON
            drive_days.model = failures.model
        ORDER BY failure_rate DESC
            
    """
    ).show(5)


#%% USING COMMON TABLE EXPRESSIONS IN SQL AND PYTHON SCOPE

# SQL
spark.sql(
    """
    WITH drive_days AS (
        SELECT
            model,
            COUNT(*) AS drive_days
        FROM
            drive_stats
        GROUP BY
            model),
    
        failures AS (
        SELECT
            model,
            COUNT(*) AS failures
        FROM
            drive_stats
        WHERE
            failure = 1
        GROUP BY
            model)
    
    SELECT
        failures.model,
        failures / drive_days AS failure_rate
    FROM
        drive_days
    INNER JOIN
        failures
    ON
        drive_days.model = failures.model
    ORDER BY
        drive_days DESC
            
    """
    ).show(5)

# PYSPARK / PYTHON
def failure_rate(drive_stats):
    drive_days = drive_stats.groupby(F.col("model")).agg(
        F.count(F.col("*")).alias("drive_days")
    )
    failures = (
        drive_stats.where(F.col("failure") == 1)
        .groupby(F.col("model"))
        .agg(F.count(F.col("*")).alias("failures"))
    )
    answer = (
        drive_days.join(failures, on="model", how="inner")
        .withColumn("failure_rate", F.col("failures") / F.col("drive_days"))
        .orderBy(F.col("failure_rate").desc())
    )
    return answer

failure_rate(backblaze_2019).show(5)


#%% OPTIMIZING THE PIPELINE
# delete variables
for v in dir():
    exec('del '+ v)
    del v
    
import os
os.chdir("/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/chapter 7")
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException # exceptions
import pyspark.sql.functions as F
import pyspark.sql.types as T
from functools import reduce
    
spark = SparkSession.builder.getOrCreate()

data_dir = "/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/data" \
    "/DataAnalysisWithPythonAndPySpark-Data-trunk/backblaze"
    
data_files = [
    "data_Q1_2019",
    "data_Q2_2019",
    "data_Q3_2019",
    "data_Q4_2019"
    ]

data = [
        spark.read.csv(os.path.join(data_dir, file),
                       header=True,
                       inferSchema=True)
        for file in data_files]

commom_cols = list(
    reduce(lambda x, y: x.intersection(y), [set(df.columns) for df in data]))

full_data = reduce(
    lambda x, y: x.select(commom_cols).union(y.select(commom_cols)), data)

#%% SQL STYLE CODE USING selectEXPR
full_data = full_data.selectExpr(
    "model", "capacity_bytes / POW(1024, 3) AS capacity_GB" ,
    "date", "failure")

drive_days = full_data.groupBy("model", "capacity_GB").agg(
    F.count("*"))

failures = (
    full_data.where("failure = 1")
    .groupBy("model", "capacity_GB")
    .agg(F.count("*").alias("failures")))

summarized_data = (
    drive_days.join(failures, on=["model", "capacity_GB"], how="left")
    .fillna(0.0, ["failures"])
    .selectExpr("model", "capacity_GB", "failures / drive_days AS failure_rate")
    .cache()
    )

# p174


