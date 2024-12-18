import os
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T 
import pandas as pd
from typing import Optional
from pyspark.ml.feature import VectorAssembler, Imputer, MinMaxScaler
from pyspark.ml.stat import Correlation


#%% DATA IMPORT
spark = (
    SparkSession.builder.appName("Recipes ML model - Are you a dessert?")
    .config("s park.driver.memory", "4gb")
    .getOrCreate()
    )

food_dir = "/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/data/DataAnalysisWithPythonAndPySpark-Data-trunk"

food = spark.read.csv(
    os.path.join(food_dir, "recipes", "epi_r.csv"),
    inferSchema=True,
    header=True
    )

print(food.count(), len(food.columns))
food.printSchema()


# STANDARDIZING COLUMN NAMES toDF()
def sanitize_column_name(name):
    """ drops unwanted characters """
    answer = name
    for i, j in ((" ", "_"), ("-", "_"),
                 ("/", "_"), ("&", "and")):
        answer = answer.replace(i, j)
    return "".join(
        [
            char
            for char in answer
            if char.isalpha() or char.isdigit() or char == "_"
        ]
    )

food = food.toDF(*[sanitize_column_name(name) for name in food.columns])

#%%
"""
PREPARING THE MACHINE LEARNING DATASET
"""


#%% DATA EXPLORATION
for x in food.columns:
    food.select(x).summary().show()
    
# IDENTIFY BINARY COLUMNS
pd.set_option("display.max_rows", 1000)

is_binary = food.agg(
    *[
          (F.size(F.collect_set(x)) == 2).alias(x)
          for x in food.columns
      ]
).toPandas()

# un-pivots a pandas dataframe
is_binary.unstack()


# INSPECT SUSPICIOUS COLUMNS
food.agg(
    *[F.collect_set(x) for x in ("cakeweek", "wasteless")]
    ).show(1, truncate=False)

food.where("cakeweek > 1.0 OR wasteless > 1.0").select(
    "title", "rating", "wasteless", "cakeweek", food.columns[-1]
).show()


# KEEPING LEGIT VALUES FOR CAKEWEEK AND WASTELESS
food = food.where(
    (
         F.col("cakeweek").isin([0.0, 1.0])
         | F.col("cakeweek").isNull()
     )
    &
    (
         F.col("wasteless").isin([0.0, 1.0])
         | F.col("wasteless").isNull()
     )
)

# lost 3 records
print(food.count(), len(food.columns))


# CATEGORIZING THE DATA SET

identifiers = ["title"]

continuous_columns = [
    "rating",
    "calories",
    "protein",
    "fat",
    "sodium"
]

target_column = ["dessert"]

binary_columns = [
    x
    for x in food.columns
    if x not in continuous_columns
    and x not in target_column
    and x not in identifiers
]


# DROP NULL VALUES
food = food.dropna(
    how="all",
    subset=[x for x in food.columns if x not in identifiers]
)

food = food.dropna(subset=target_column)

print(food.count(), len(food.columns))


# IMPUTE NULL VALUES IN BINARY COLUMNS
food = food.fillna(0.0, subset=binary_columns)

print(food.where(F.col(binary_columns[0]).isNull()).count())


# NON-NUMERICAL VALUES IN THE RATINGS AND CALORIES COLUMNS
@F.udf(T.BooleanType())
def is_a_number(value: Optional[str]) -> bool:
    if not value:
        return True
    try:
        _ = float(value)
    except ValueError:
        return False
    return True

# one bad record
food.where(~is_a_number(F.col("rating"))).select(
    *continuous_columns
).show()


# CAST RATING AND CALORIES INTO DOUBLE
for column in ["rating", "calories"]:
    food = food.where(is_a_number(F.col(column)))
    food = food.withColumn(column, F.col(column).cast(T.DoubleType()))

# one record removed
print(food.count(), len(food.columns))


# IDENTIFY OUTLIERS
food.select(*continuous_columns).summary(
    "mean",
    "stddev",
    "min",
    "1%",
    "5%",
    "50%",
    "95%",
    "99%",
    "max",
).show()


# CAP DATA AT 99TH PRECENTILE
maximum = {
    "calories": 3203.0,
    "protein": 173.0,
    "fat": 207.0,
    "sodium": 5661.0
}

for k, v in maximum.items():
    food = food.withColumn(
        k,
        F.when(F.isnull(F.col(k)), F.col(k)).otherwise(
            F.least(F.col(k), F.lit(v))
        ),
    )
    
    
# REMOVING BINARY FEATURES THAT HAPPEN TOO LITTLE OR TOO OFTEN
# count instances
inst_sum_of_binary_columns = [
    F.sum(F.col(x)).alias(x) for x in binary_columns
]
sum_of_binary_columns = (
    food.select(*inst_sum_of_binary_columns).head().asDict()
)

# filter unrelevant columns
threshold_min, threshhold_max = 10, food.count() - 10
too_rare_features = [
    k
    for k, v in sum_of_binary_columns.items()
    if v < threshold_min or v > threshhold_max
]

print(len(too_rare_features))
print(too_rare_features)

# update feature list
binary_columns = list(set(binary_columns) - set(too_rare_features))


#%% CREATING CUSTOM FEATURES
"""
FEATURE CREATION
    custom features from out continouos columns
FEATURE REFINEMENT
    measuring correlations between synthetic and true features
"""

# calories attributable to protein and fat
protein_kcal, fat_kcal = 4.0, 9.0
food = food.withColumn(
    "protein_ratio",
    F.col("protein") * protein_kcal / F.col("calories")
).withColumn(
    "fat_ratio",
    F.col("fat") * fat_kcal / F.col("calories")
)
    
food = food.fillna(0.0, subset=["protein_ratio", "fat_ratio"])

continuous_columns += ["protein_ratio", "fat_ratio"]
 
    
#%% REMOVING HIGHLY CORRELATED FEATURES
"""
- If two features are highly correlated, it means that they provide almost the same
information. In the context of machine learning, this can confuse the fitting
algorithm and create model or numerical instability.

- The more complex your model, the more complex the maintenance. Highly
correlated features rarely provide improved accuracy, yet complicate the model.
Simple is better.
"""

# assembling feature columns into a single Vector column
continuous_features = VectorAssembler(
    inputCols=continuous_columns, outputCol="continuous_features"
)

vector_food = food.select(continuous_columns)

# remove null values
for x in continuous_columns:
    vector_food = vector_food.where(~F.isnull(F.col(x)))
    
vector_variable = continuous_features.transform(vector_food)

vector_variable.select("continuous_features").show(3, truncate=False)
vector_variable.select("continuous_features").printSchema()

# calculate correlation matrix in pyspark
correlation = Correlation.corr(
    vector_variable, "continuous_features"
)

correlation.printSchema()

# calc pearson
correlation_array = correlation.head()[0].toArray()

# convert to pandas
correlation_pd = pd.DataFrame(
    correlation_array,
    index=continuous_columns,
    columns=continuous_columns
)

print(correlation_pd.iloc[:, :4])
print(correlation_pd.iloc[:, 4:])


#%% FEATURE PREPARATION WITH TRANSFORMERS AND ESTIMATORS
# create a data imputer
old_cols = ["calories", "protein", "fat", "sodium"]
new_cols = ["calories_i", "protein_i", "fat_i", "sodium_i"]

# create imputer instance
imputer = Imputer(
    strategy="mean",
    inputCols=old_cols,
    outputCols=new_cols
)

# create imputer model
imputer_model = imputer.fit(food)

continuous_columns = (
    list(set(continuous_columns) - set(old_cols)) + new_cols
)

# impute the data
food_imputed = imputer_model.transform(food)

# check the imputed results
food_imputed.where("calories IS null").select("calories",
                                              "calories_i").show(5)

#%% SCALE THE DATA WITH MIN MAX SCALER
continuous_nb = [x for x in continuous_columns if "ratio" not in x]

# create a vector from the columns
continuous_assembler = VectorAssembler(
    inputCols=continuous_nb, outputCol="continuous"
)
food_features = continuous_assembler.transform(food_imputed)

# create the min max scaler
continuous_scaler = MinMaxScaler(
    inputCol="continuous",
    outputCol="continuous_scaled"
)

food_features = continuous_scaler.fit(
    food_features).transform(
        food_features
)

food_features.select("continuous_scaled").show(3, False)