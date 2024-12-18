import os
os.chdir("/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/chapter 9")
from functools import reduce
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd
import pyspark.sql.types as T
from time import sleep
from typing import Iterator, Tuple

spark = SparkSession.builder.getOrCreate()


#%% COLUMN TRANSFORMATIONS WITH PANDAS: USING SERIES UDF
"""
typical steps:
    install and configure the connector
    customize the SparkReader object
    Read data, authenticating as needed
"""

# google BigQuery p195


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


gsod.show(5)


#%% CREATING PANDAS UDF THAT TRANSFORMS FAHRENHEIT TO CELSIUS (scalar)

# create the pandas UDF
@F.pandas_udf(T.DoubleType())
def f_to_c(degrees: pd.Series) -> pd.Series:
    """ transform farenheit to celsius """
    return (degrees - 32) * 5 / 9

# apply the UDF
gsod = gsod.withColumn(
    "temp_c",
    f_to_c(F.col("temp"))
)

gsod.select("temp", "temp_c").distinct().show(5)


#%% USING AN ITERATOR OF SERIES TO ITERATOR OF SERIES UDF (scalar)
""" used with expensive cold start """

@F.pandas_udf(T.DoubleType())
def f_to_c2(degrees: Iterator[pd.Series]) -> Iterator[pd.Series]:
    """ transforms f to c """
    sleep(5) # simulated cold start
    # iterate over each batch
    for batch in degrees:
        yield (batch - 32) * 5 / 9
        
gsod.select(
    "temp",
    f_to_c2(F.col("temp")).alias("temp_c")
    ).distinct().show(5)
        

#%% USING AN ITERATOR OF MULTIPLE SERIES UDF (series)

@F.pandas_udf(T.DateType())
def create_date(
        year_mo_da: Iterator[Tuple[pd.Series, pd.Series, pd.Series]]
        ) -> Iterator[pd.Series]:
    """ merge three cols into a date col """
    for year, mo, da in year_mo_da:
        yield pd.to_datetime(
            pd.DataFrame(dict(year=year, month=mo, day=da))
        )

gsod.select(
    "year", "mo", "da",
    create_date(F.col("year"), F.col("mo"), F.col("da")).alias("date")
    ).distinct().show(5)
