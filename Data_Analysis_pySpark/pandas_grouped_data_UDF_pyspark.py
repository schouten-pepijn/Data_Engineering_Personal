import os
os.chdir("/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/chapter 9")
from sklearn.linear_model import LinearRegression
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd
import pyspark.sql.types as T
from functools import reduce

spark = SparkSession.builder.getOrCreate()


"""
split-apply-combine pattern:
    Split your data into logical batches (groupBy)
    Apply a function to each batch independently
    Combine the batches into a unified data set
"""

#%% READ DATA LOCALLY

data_dir = "/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/data" \
    "/DataAnalysisWithPythonAndPySpark-Data-trunk/gsod_noaa"
    
gsod = (
        reduce(
            lambda x, y: x.unionByName(y, allowMissingColumns=True),
            [
                spark.read.parquet(data_dir + f"/gsod{year}.parquet")
                for year in range(2010, 2021)
            ],
        )
        .dropna(subset=["year", "mo", "da", "temp"])
        .where(F.col("temp") != 999.9)
        .drop("date")
)


#%% GROUP AGGREGATE UDFs (returns scalar)
@F.pandas_udf(T.DoubleType())
def rate_of_change_temp(day: pd.Series, temp: pd.Series) -> float:
    """ returns the slope of the daily temperature for a given period of time """
    return (
        LinearRegression()
        .fit(X=day.astype(int).values.reshape(-1, 1), y=temp)
        .coef_[0])

result = gsod.groupBy("stn", "year", "mo").agg(
    rate_of_change_temp(gsod["da"], gsod["temp"]).alias(
        "rate_change_temp")
)

result.show(5, truncate=False)


#%% GROUP MAP UDF (returns dataframe)
def scale_temp(temp_by_day: pd.DataFrame) -> pd.DataFrame:
    """ returns a simple normalization of the temperature of a site """
    temp = temp_by_day.temp
    answer = temp_by_day[["stn", "year", "mo", "da", "temp"]]
    if temp.min() == temp.max():
        return answer.assign(temp_norm=0.5)
    else:
        return answer.assign(
            temp_norm=(temp - temp.min()) / (temp.max() - temp.min()))
    
# apply the udf
gsod_map = gsod.groupBy("stn", "year", "mo").applyInPandas(
    func=scale_temp,
    schema=(
        "stn string, year string, mo string, "
        "da string, temp double, temp_norm double"
        ),
)

gsod_map.show(5, truncate=False)


#%% TESTING UDF ON A SAMPLE
gsod_local = gsod_map.where(
    "year = '2018' and mo = '08' and stn = '710920'"
    ).toPandas()

print(rate_of_change_temp.func(
      gsod_local["da"], gsod_local["temp_norm"]
      )
)


#%% WHEN TO USE WHICH
"""
- If you need to control how the batches are made, you need to use a grouped
data UDF. If the return value is scalar, group aggregate, or otherwise, use a
group map and return a transformed (complete) data frame.

- If you only want batches, you have more options. The most flexible is mapIn-
Pandas(), where an iterator of pandas DataFrame comes in and a transformed
one comes out. This is very useful when you want to distribute a pandas/local
data transformation on the whole data frame, such as with inference of local
ML models. Use it if you work with most of the columns from the data frame,
and use a Series to Series UDF if you only need a few columns.

- If you have a cold-start process, use a Iterator of Series/multiple Series UDF,
depending on the number of columns you need within your UDF.

 - If you only need to transform some columns using pandas, a Series to
Series UDF is the way to go.
"""
