import os
os.chdir("/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/chapter 13")
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T 
from typing import Optional
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
import pyspark.ml.feature as MF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator


#%% DATA IMPORT
spark = (
    SparkSession.builder.appName("Recipes ML model - Are you a dessert?")
    .config("s park.driver.memory", "6gb")
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

# IMPUTE NULL VALUES IN BINARY COLUMNS
food = food.fillna(0.0, subset=binary_columns)


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

# CAST RATING AND CALORIES INTO DOUBLE
for column in ["rating", "calories"]:
    food = food.where(is_a_number(F.col(column)))
    food = food.withColumn(column, F.col(column).cast(T.DoubleType()))

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

# update feature list
binary_columns = list(set(binary_columns) - set(too_rare_features))


#%% CREATING CUSTOM FEATURES
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
 

#%% EXPLOEING THE TRANSFORMER METHOD
continuous_nb = [x for x in continuous_columns if "ratio" not in x]

# create a vector from the columns
continuous_assembler = VectorAssembler(
    inputCols=continuous_nb, outputCol="continuous"
)

# create the min max scaler
continuous_scaler = MinMaxScaler(
    inputCol="continuous",
    outputCol="continuous_scaled"
)

# accssing transformer parameters to yield a Param instance
print(continuous_assembler.outputCol)
# accessing the outputCol values
print(continuous_assembler.getOutputCol())
# return all Param values
print(continuous_assembler.explainParams())

#%% MODIFYING PARAMS IN A TRANSFORMER WITH SETTERS
"""
- You are building your transformer in the REPL, and you want to experiment
with different Param-eterizations.
- You are optimizing your ML pipeline Params
"""

# change one param
continuous_assembler.setOutputCol("more_continuous")
print(continuous_assembler.getOutputCol())

# change multiple params
continuous_assembler.setParams(
    inputCols=["one", "two", "three"], handleInvalid="skip")
print(continuous_assembler.explainParams())

# clearing current value of the handleInvalid Param with clear()
continuous_assembler.clear(continuous_assembler.handleInvalid)
print(continuous_assembler.getHandleInvalid())

# linked and unlinked transformers
# linked
new_continuous_assembler = continuous_assembler
new_continuous_assembler.setOutputCol("new_input")
print(new_continuous_assembler.getOutputCol())
print(continuous_assembler.getOutputCol())

# unlinked
copy_continuous_assembler = continuous_assembler.copy()
copy_continuous_assembler.setOutputCol("copy_input")
print(copy_continuous_assembler.getOutputCol())
print(continuous_assembler.getOutputCol())


#%% BUiLDING A COMPLETE ML PIPELINE
"""
Using pipeline class
"""

# create the imputer
imputer = MF.Imputer(
    strategy="mean",
    inputCols=["calories", "protein", "fat", "sodium"],
    outputCols=["calories_i", "protein_i", "fat_i", "sodium_i"] 
)

# create the vector assembler
continuous_assembler = MF.VectorAssembler(
    inputCols=["rating", "calories_i", "protein_i", "fat_i", "sodium_i"],
    outputCol="continuous"
)

# create the min max scaler
continuous_scaler = MF.MinMaxScaler(
    inputCol="continuous",
    outputCol="continuous_scaled"
)

# create the pipeline
food_pipeline = Pipeline(
    stages=[imputer, continuous_assembler, continuous_scaler]    
)

# create the features for the ML model
preml_assembler = MF.VectorAssembler(
    inputCols=binary_columns
    + ["continuous_scaled"]
    + ["protein_ratio", "fat_ratio"],
    outputCol="features"    
)

# change the stages of the pipeline
food_pipeline.setStages(
    [imputer, continuous_assembler, continuous_scaler, preml_assembler]    
)

# create pipeline model
food_pipeline_model = food_pipeline.fit(food)
food_features = food_pipeline_model.transform(food)


#%% INVESTIGATING THE PIPELINE MODEL
# displaying the features
food_features.select(
    "title", "dessert", "features").show(5, truncate=30)

# reading the metadata
print(food_features.schema["features"])
print(food_features.schema["features"].metadata)


#%% TRAINING THE LOGISTIC REGRESSION MODEL
# create logistic regression model
log_reg = LogisticRegression(
    featuresCol="features", labelCol="dessert", predictionCol="prediction"
)

# update the pipeline
food_pipeline.setStages(
    [
         imputer, 
         continuous_assembler, 
         continuous_scaler, 
         preml_assembler,
         log_reg
     ]
)

# split in train / test split
train, test = food.randomSplit([0.7, 0.3], 1234)
train.cache()

# train the model
food_pipeline_model = food_pipeline.fit(train)

# make predictions
results = food_pipeline_model.transform(test)

# show predictions
results.select(
    "prediction", "rawPrediction", "probability").show(3, False)


#%% EVALUATING THE LOGISTIC REGRESSION MODEL
# create a confusion matrix using pivot()
results.groupBy("dessert").pivot("prediction").count().show()

# computing the precision and recall
log_reg_model = food_pipeline_model.stages[-1]  # last stage of pipeline
metrics = log_reg_model.evaluate(
    results.select("title", "dessert", "features"))

# print precision and recall
print(f"Model precision: {metrics.precisionByLabel[1]}")
print(f"Model recall: {metrics.recallByLabel[1]}")


# calculating the AUC
evaluator = BinaryClassificationEvaluator(
    labelCol="dessert",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(results)
print(f"Area under ROC: {auc}")

# display the ROC curve - matplotlib
plt.figure(dpi=100, tight_layout=True, figsize=(5, 5))
plt.plot([0, 1], [0, 1], "r--")
plt.plot(
    log_reg_model.summary.roc.select("FPR").collect(),
    log_reg_model.summary.roc.select("TPR").collect()
)
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.show()


#%% OPTIMIZING THE MODEL
# build a set of hyperparameters in a grid
grid_search = (
    ParamGridBuilder()
    .addGrid(log_reg.elasticNetParam, [0.0, 0.5, 1.0])
    .build()
)
print(grid_search)

# creating and using a k-folds crossvalidation object
cv = CrossValidator(
    estimator=food_pipeline,
    estimatorParamMaps=grid_search,
    evaluator=evaluator,
    numFolds=3,
    seed=1234,
    collectSubModels=True
)

# fit the model
cv_model = cv.fit(train)

print(cv_model.avgMetrics)

pipeline_food_model = cv_model.bestModel

# extracting the feature names from the features vector
# create column names
feature_names = ["(Intercept)"] + [
    x["name"]
    for x in (
            food_features
            .schema["features"]
            .metadata["ml_attr"]["attrs"]["numeric"]
    )
]

# create intercept and slope values
feature_coefficients = [log_reg_model.intercept] + list(
    log_reg_model.coefficients.values
)

# create dataframe
coefficients = pd.DataFrame(
    feature_coefficients, index=feature_names, columns=["coef"]
)
coefficients["abs_coef"] = coefficients["coef"].abs()

print(coefficients.sort_values(["abs_coef"]))


#%% SAVE THE MODEL
pipeline_food_model.write().overwrite().save("am_I_a_dessert_the_model")


#%% LOAD THE MODEL
loaded_model = PipelineModel.load("am_I_a_dessert_the_model")
