import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from fractions import Fraction
from typing import Tuple, Optional


spark = SparkSession.builder.getOrCreate()

#%% CREATE FRACTION DATA SET
fractions = [[x, y] for x in range(100) for y in range(1, 100)]

frac_df = spark.createDataFrame(
    fractions,
    ["numerator", "denominator"]
)

frac_df = frac_df.select(
    F.array(F.col("numerator"), F.col("denominator")).alias(
        "fraction"),
)

frac_df.show(5, truncate=False)

#%% CREATING PYTHON UDFs
"""
Hints:
    Create and document the function
    Make sure the input and output types are compatible
    Test the function
"""

# create a type synonym
Frac = Tuple[int, int]

# create reduction function
def py_reduce_fraction(frac: Frac) -> Optional[Frac]:
    """ reduce fraction (2-tuple of ints) """
    num, denom = frac
    if denom:
        answer = Fraction(num, denom)
        return answer.numerator, answer.denominator
    else:
        return None
    
assert py_reduce_fraction((3, 6)) == (1, 2)
assert py_reduce_fraction((1, 0)) == None

# create transformation function
def py_fraction_to_float(frac: Frac) -> Optional[float]:
    """ transforms a fraction (2-tuple of ints) into a float """
    num, denom = frac
    if denom:
        return num / denom
    else:
        return None
    
assert py_fraction_to_float((2, 8)) == 0.25
assert py_fraction_to_float((1, 0)) == None

#%% CREATING EXPLICIT UDFS with udf()
SparkFrac = T.ArrayType(T.LongType())

# create udf
reduce_fraction = F.udf(py_reduce_fraction, SparkFrac)

# apply udf
frac_df = frac_df.withColumn(
    "reduced_fraction",
    reduce_fraction(F.col("fraction"))
)

frac_df.show(5, truncate=False)

#%% CREATE DECORATOR UDF with @F.udf


# create decorated udf function
@F.udf
def fraction_to_float(frac: Frac) -> Optional[float]:
    """ transforms a fraction (2-tuple of ints) into a float """
    num, denom = frac
    if denom:
        return num / denom
    else:
        return None

# apply udf
frac_df = frac_df.withColumn(
    "fraction_float",
    fraction_to_float(F.col("fraction"))
)

# show distinct values
frac_df.select("reduced_fraction", "fraction_float").distinct().show(
    5, truncate=False)

assert fraction_to_float.func((1, 2)) == 0.5


