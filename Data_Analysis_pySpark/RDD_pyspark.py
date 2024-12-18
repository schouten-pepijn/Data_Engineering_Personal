from pyspark.sql import SparkSession
from py4j.protocol import Py4JJavaError
from operator import add

#%% PROMOTE PYTHON FUNCTION TO AN RDD

spark = SparkSession.builder.getOrCreate()

collection = [1, "two", 3.0, ("four", 4), {"five": 5}]

sc = spark.sparkContext # holds RDD functions and methods

# promote collecion to RDD
collection_rdd = sc.parallelize(collection)

print(collection_rdd)

#%% MAPPING A FUNCTION TO EACH ELEMENT (FAIL)
def add_one(value):
    return value + 1

collection_rdd = collection_rdd.map(add_one)

try:
    print(collection_rdd.collect())
except Py4JJavaError:
    pass

#%% MAPPING A FUNCTION TO EACH ELEMENT (SAFER)
collection_rdd = sc.parallelize(collection)

def safer_add_one(value):
    try:
        return value + 1
    except TypeError:
        return value
    
collection_rdd = collection_rdd.map(safer_add_one)

print(collection_rdd.collect())

#%% FILTERING THE RDD WITH A LAMBDA FUNCTION
collection_rdd = collection_rdd.filter(
    lambda elem: isinstance(elem, (float, int))
)

print(collection_rdd.collect())

#%% REDUCING ELEMENTS IN RDD WITH REDUCE
collection_rdd = sc.parallelize([4, 7, 9, 1, 3])

print(collection_rdd.reduce(add))
