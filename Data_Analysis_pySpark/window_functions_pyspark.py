import os
os.chdir("/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/chapter 10")
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import pandas as pd

spark = SparkSession.builder.getOrCreate()

data_dir = "/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/" \
    "data/DataAnalysisWithPythonAndPySpark-Data-trunk/window/gsod.parquet"
gsod = spark.read.parquet(data_dir)


#%% COLDEST DAY OF EACH YEAR - old way
coldest_temp = gsod.groupBy("year").agg(
    F.min("temp").alias("temp"))
coldest_temp.orderBy("temp").show()

coldest_when = gsod.join(
    coldest_temp, how="left_semi", on=["year", "temp"]
    ).select("stn", "year", "mo", "da", F.col("temp").alias("min_temp"))
coldest_when.orderBy("year", "mo", "da").show()


#%% USING WINDOW FUNCTIONS - performance increase on big data sets
"""
Window functions vocabulary:
    Partition the data frame
    Select values over the window
    Combine and union is inplicit
"""

# create window
each_year = Window.partitionBy("year")
print(each_year)

gsod.withColumn(
    "min_temp", F.min("temp").over(each_year) # apply the window
    ).where("temp = min_temp"
    ).select("year", "mo", "da", "stn", F.col("temp").alias("min_temp")
    ).orderBy("year", "mo", "da"
    ).show()

# using window function within a select() methods
gsod.select(
    "year", "mo", "da", "stn","temp",
    F.min("temp").over(each_year).alias("min_temp"),
    ).where(
        "temp = min_temp"
    ).drop("temp"
    ).orderBy("year", "mo", "da"
    ).show()
              
              
#%% RANKING FUNCTIONS
data_dir = "/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/data" \
    "/DataAnalysisWithPythonAndPySpark-Data-trunk/window/gsod_light.parquet"
gsod_light = spark.read.parquet(data_dir)

gsod_light.show()

# an ordered window function
temp_per_month_asc = Window.partitionBy("mo").orderBy("count_temp")

# ranking on count_temp col with ties
gsod_light.withColumn(
    "rank_tpm",
    F.rank().over(temp_per_month_asc)
).show()
    
# ranking on count_temp without ties
gsod_light.withColumn(
    "rank_tpm",
    F.dense_rank().over(temp_per_month_asc)
).show()

# percent rank
temp_each_year = Window.partitionBy("year").orderBy("temp")
gsod_light.withColumn(
    "rank_tpm",
    F.percent_rank().over(temp_each_year)
).show()

# creating buckets over ranks - two-tile value over the window
gsod_light.withColumn(
    "temp_tpm",
    F.ntile(2).over(temp_each_year)
).show()

# numbering records within each window
gsod_light.withColumn(
    "temp_tpm",
    F.row_number().over(temp_each_year)
).show()


# descending-ordered column
temp_per_month_desc = Window.partitionBy("mo").orderBy(
    F.col("count_temp").desc()
)

gsod_light.withColumn(
    "row_number",
    F.row_number().over(temp_per_month_desc)
).show()


#%% ANALYTICAL FUNCTIONS

# previous two values using lag()
gsod_light.withColumn(
    "previous_temp", F.lag("temp").over(temp_each_year)
).withColumn(
    "previous_temp_2", F.lag("temp", 2).over(temp_each_year)
).show()
    
# cumulative distribution over window
gsod_light.withColumn(
    "percent_rank",
    F.percent_rank().over(temp_each_year)
).withColumn(
    "cume_dist",
    F.cume_dist().over(temp_each_year)
).show()


#%% RANGING AND BOUNDING WINDOWS

# ordering and not ordering defaults to different window functions
not_ordered = Window.partitionBy("year")
ordered = Window.partitionBy("year").orderBy("temp")

# different results
gsod_light.withColumn(
    "avg_NO",
    F.avg("temp").over(not_ordered)
).withColumn(
    "avg_O",
    F.avg("temp").over(ordered)
    ).show(5)
    
# setting window boundaries
not_ordered = Window.partitionBy("year").rowsBetween(
    start=Window.unboundedPreceding,
    end=Window.unboundedFollowing)

ordered = not_ordered.orderBy("temp").rangeBetween(
    start=Window.unboundedPreceding,
    end=Window.currentRow)

# creating a data frame with date column to apply range window on
gsod_light_p = (
    gsod_light.withColumn("year", F.lit(2019))
    .withColumn(
        "dt", 
        F.to_date(
            F.concat_ws("-", F.col("year"), F.col("mo"), F.col("da"))
        ),
    )
    .withColumn("dt_num", F.unix_timestamp("dt"))
)
gsod_light_p.show()

# usings a 60 day sliding range window to compute the average
one_month_ish = 30 * 60 * 60 * 24
one_month_ish_before_and_after = (
    Window.partitionBy("year")
    .orderBy("dt_num")
    .rangeBetween(-one_month_ish, one_month_ish)
)

gsod_light_p.withColumn(
    "avg_count",
    F.avg("count_temp").over(one_month_ish_before_and_after)
).show()


# using UDFs within windows
@F.pandas_udf("double")
def median_udf(vals: pd.Series) -> float:
    return vals.median()

# unbounded median versus right-bounded median
gsod_light.withColumn(
    "median_temp", median_udf("temp").over(
        Window.partitionBy("year"))
    ).withColumn(
        "median_temp_g",
        median_udf("temp").over(
            Window.partitionBy("year").orderBy("mo", "da")
        ),
    ).show()
        
        
"""
Tip for window functions:
    - What kind of operation do I want to perform? Summarize, rank, or look
      ahead/behind.
    - How do I need to construct my window? Should it be bounded or unbounded?
      Do I need every record to have the same window value (unbounded), or should
      the answer depend on where the record fits within the window (bounded)?
      When bounding a window frame, you most often want to order it as well.
    - For bounded windows, do you want the window frame to be set according to the
      position of the record (row based) or the value of the record (range based)?
    - Finally, remember that a window function does not make your data frame spe-
      cial. After your function is applied, you can filter, group by, and even apply
      another, completely different, window.
"""