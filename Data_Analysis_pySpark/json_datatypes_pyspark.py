import os
os.chdir("/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/chapter 5/part 2")
import json
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pprint import pprint

"""
json
"""

sample_json = """{
    "id": 143,
    "name": "Silicon Valley",
    "type": "Scripted",
    "language": "English",
    "genres": [
        "Comedy"
    ],
    "network": {
        "id": 8,
        "name": "HBO",
        "country": {
            "name": "United States",
            "code": "US",
            "timezone": "America/New_York"
        }
    }
}"""


# load json file using native python
document = json.loads(sample_json)

print(document)
print(type(document))

# reading json file using pyspark
base_path = "/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/data/" \
    "DataAnalysisWithPythonAndPySpark-Data-trunk/shows"
spark = SparkSession.builder.getOrCreate()
shows = spark.read.json(os.path.join(base_path, "shows-*.json"))

print(f"number of records: {shows.count()}")

shows.printSchema()

print(f"column names: {shows.columns}")

# select subsection of data frame to show array structure
array_subset = shows.select(F.col("name"), F.col("genres"))
array_subset.show(10, truncate=False)

""" 
arrays
"""

# extract elements from array
array_subset = array_subset.select(
    F.col("name"),
    array_subset.genres[0].alias("dot_and_index"),
    F.col("genres")[0].alias("col_and_index"),
    array_subset.genres.getItem(0).alias("dot_and_method"),
    F.col("genres").getItem(0).alias("col_and_method")
    )
array_subset.show()

# performing operations on arrays
array_subset_repeated = array_subset.select(
    F.col("name"),
    F.lit("Comedy").alias("one"),
    F.lit("Horror").alias("two"),
    F.lit("Drama").alias("three"),
    F.col("dot_and_index"),
    ).select(
        F.col("name"),
        F.array("one", "two", "three").alias("some_genres"),
        F.array_repeat(F.col("dot_and_index"), 5).alias("repeated_genres")
    )
        
# compute elements in arrays
array_subset_repeated.select(
    F.col("name"),
    F.size(F.col("some_genres")),
    F.size(F.col("repeated_genres"))
    ).show()

# Remove duplicates in arrays
array_subset_repeated.select(
    F.col("name"),
    F.array_distinct(F.col("some_genres")),
    F.array_distinct(F.col("repeated_genres"))
    ).show(truncate=False)

# intercecting two arrays
array_subset_repeated = array_subset_repeated.select(
    F.col("name"),
    F.array_intersect(F.col("some_genres"), F.col("repeated_genres")
                      ).alias("genres"),
    )
array_subset_repeated.show()

# search for positions
array_subset_repeated.select(
    F.col("genres"),
    F.array_position(F.col("genres"), "Comedy"
                     ).alias("comedy_index")
    ).show()

"""
maps - typed dicts
"""

columns = ["name", "language", "type"]

# create a map from two arrays
shows_map = shows.select(
    *[F.lit(column) for column in columns],
    F.array(*columns).alias("values"),
)

shows_map = shows_map.select(
    F.array(*columns).alias("keys"), F.col("values"))

shows_map.show()

shows_map = shows_map.select(
    F.map_from_arrays(F.col("keys"), F.col("values")).alias("mapped"))

shows_map.printSchema()
shows_map.show()

# acces values by keys
shows_map.select(
    F.col("mapped.name"),
    F.col("mapped")["name"],
    shows_map.mapped["name"],
    ).show()


"""
strucs - size known beforehand
"""

shows.select(F.col("schedule")).printSchema()
shows.select(F.col("_embedded")).printSchema()

# promoting fields in struc to columns
shows_clean = shows.withColumn(
    "episodes", F.col("_embedded")["episodes"]
    ).drop("_embedded")

shows_clean.printSchema()

# selecting a field in an Array[sStruct] to create a column
episodes_name = shows_clean.select(F.col("episodes.name"))
episodes_name.printSchema()

episodes_name.select(F.explode(F.col("name")
                               ).alias("name")
                     ).show(truncate=False)


"""
creating a schema with StructField() ValueType()
"""

episode_links_schema = T.StructType(
    [
         T.StructField(
             "self", T.StructType([T.StructField("href", T.StringType())])
         )    
     ]
)

episode_image_schema = T.StructType(
    [
         T.StructField("medium", F.StringType()),
         T.StructField("original", F.StringType())
     ]
)


episode_schema = T.StructType(
    [
         T.StructField("_links", episode_links_schema),
         T.StructField("airdate", T.DateType()),
         T.StructField("airstamp", T.TimestampType()),
         T.StructField("airtime", T.StringType()),
         T.StructField("id", T.StringType()),
         T.StructField("image", episode_image_schema),
         T.StructField("name", T.StringType()),
         T.StructField("number", T.LongType()),
         T.StructField("runtime", T.LongType()),
         T.StructField("season", T.LongType()),
         T.StructField("summary", T.StringType()),
         T.StructField("url", T.StringType()),
    ]
)

embedded_schema = T.StructType(
    [
         T.StructField(
             "_embedded",
             T.StructType(
                 [
                     T.StructField(
                         "episodes", T.ArrayType(episode_schema)
                     )
                 ]
             )
         )
     ]
)
      

"""
reading json with strict schema
"""

shows_with_schema = spark.read.json(
    os.path.join(base_path, "shows-silicon-valley.json"),
    schema=embedded_schema,
    mode="FAILFAST")

# validation
for col in ["airdate", "airstamp"]:
    shows.select(f"_embedded.episodes.{col}").select(
        F.explode(col)).show(5)    


"""
creating schemas with JSON
"""
pprint(
       shows_with_schema.select(
           F.explode(F.col("_embedded.episodes")).alias("episode")
       )
        .select(F.col("episode.airtime"))
        .schema.jsonValue()
)

# seralizing schema as json
other_shows_schema = T.StructType.fromJson(
    json.loads(shows_with_schema.schema.json()))

pprint(other_shows_schema)
print(other_shows_schema == shows_with_schema.schema)


"""
from hierarchical to tabular data
"""

# exploding the _embedded.episodes into distinct records
episodes = shows.select(
    F.col("id"),
    F.explode("_embedded.episodes").alias("episodes")
)
episodes.show(5, truncate=70)
print(f"record count: {episodes.count()}")

# exploding a map with posexplode()
episode_name_id = shows.select(
    F.map_from_arrays(
        F.col("_embedded.episodes.id"), F.col("_embedded.episodes.name")
    ).alias("name_id")
)

episode_name_id = episode_name_id.select(
    F.posexplode(F.col("name_id")).alias("position", "id", "name")
)

episode_name_id.show(5)

# collecting results back in array
collected = episodes.groupBy(F.col("id")).agg(
    F.collect_list(F.col("episodes")).alias("episodes")
)
print(f"collected count: {collected.count()}")
collected.printSchema()

"""
creating struct field in data frame
"""
struct_ex = shows.select(
    F.struct(
        F.col("status"), F.col("weight"), F.lit(True).alias("has_watched")
    ).alias("info")
)

struct_ex.show(n=50, truncate=False)
struct_ex.printSchema()