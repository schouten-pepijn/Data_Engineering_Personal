import os 
os.chdir("/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/chapter 5")

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
 
spark = SparkSession.builder.getOrCreate()

base_dir = "/Users/pepijnschouten/Desktop/Data_Analysis_Pyspark/data/DataAnalysisWithPythonAndPySpark-Data-trunk/broadcast_logs"

logs = spark.read.csv(
    os.path.join(base_dir, "BroadcastLogs_2018_Q3_M8_sample.csv"),
    sep="|",
    header=True,
    inferSchema=True,
    timestampFormat="yyyy-MM-dd")

logs = logs.withColumn(
    "duration_seconds",
    (
         F.col("Duration").substr(1, 2).cast("int") * 60 * 60
         + F.col("Duration").substr(4, 2).cast("int") * 60
         + F.col("Duration").substr(7, 2).cast("int")
     ),
)

log_identifier = spark.read.csv(
    os.path.join(base_dir, "ReferenceTables", "LogIdentifier.csv"),
    sep="|",
    header=True,
    inferSchema=True,)

log_identifier.printSchema()

""" filter to keep primary channels """
log_identifier = log_identifier.filter(F.col("PrimaryFG") == 1)
print(log_identifier.count())
log_identifier.show(5)

""" Syntax of left join
[LEFT].join(
[RIGHT],
on=[PREDICATES]
how=[METHOD]
)
"""

""" bare bone left join of two tables """
logs.join(
    log_identifier,
    on="LogServiceID",
    how="inner"
    ).show()


""" identical column names """
logs_and_channels_verbose = logs.join(
    log_identifier, logs["LogServiceID"] == log_identifier["LogServiceID"])

logs_and_channels_verbose.printSchema()
try:
    logs_and_channels_verbose.select("LogSerivceID")
except:
    print("error, double column names")
    
    
""" solution one, combines the column"""
logs_and_channels = logs.join(
    log_identifier, on="LogServiceID", how="inner")
logs_and_channels.printSchema()

""" solution two, by remembering column origin """
logs_and_channels_verbose = logs.join(
    log_identifier,
    on=logs["LogServiceID"] == log_identifier["LogServiceID"],
    how="inner")
# drop one column
logs_and_channels_verbose.drop(log_identifier["LogServiceID"]).select(
    "LogServiceID").show(1)

""" solution three, by using aliasses method """
logs_and_channels_verbose = logs.alias("left").join(
    log_identifier.alias("right"),
    on=logs["LogServiceID"] == log_identifier["LogServiceID"],
    how="inner")
logs_and_channels_verbose.drop(F.col("right.LogServiceID")).select(
    "LogServiceID").show(1)


""" perform the other joins by chaining """
cd_category = spark.read.csv(
    os.path.join(base_dir, "ReferenceTables", "CD_Category.csv"),
    sep="|",
    header=True,
    inferSchema=True).select(
        "CategoryID", "CategoryCD",
        F.col("EnglishDescription").alias("Category_Description"))
        
cd_program_class = spark.read.csv(
    os.path.join(base_dir, "ReferenceTables", "CD_ProgramClass.csv"),
    sep="|",
    header=True,
    inferSchema=True).select(
        "ProgramClassID", "ProgramClassCD",
        F.col("EnglishDescription").alias("ProgramClass_Description"))
        
full_log = logs_and_channels.join(
    cd_category,
    on="CategoryID",
    how="left").join(
        cd_program_class,
        on="ProgramClassID",
        how="left")

full_log.printSchema()
        
""" perform a groupby for duration length """
duration_length = full_log.groupBy(
    "ProgramClassID", "ProgramClass_Description").agg(
        F.sum("duration_seconds").alias("duration_total")).orderBy(
            "duration_total", ascending=False)
            
duration_length.show(60, truncate=False)        


""" grouping and aggregating on custom columns """

""" blueprint
(
F.when([BOOLEAN TEST], [RESULT IF TRUE])
.when([ANOTHER BOOLEAN TEST], [RESULT IF TRUE])
.otherwise([DEFAULT RESULT, WILL DEFAULT TO null IF OMITTED])
)
"""
answer = (
    full_log.groupBy("LogIdentifierID")
    .agg(
        F.sum(
            F.when(
                F.trim(F.col("ProgramClassID")).isin(
                    ["COM", "PRC", "PGI", "PRO", "LOC", "SPO", "MER", "SOL"]
                ),
                F.col("duration_seconds"),
            ).otherwise(0)
        ).alias("duration_commercial"),
        F.sum("duration_seconds").alias("duration_total"),
    )
    .withColumn("commercial_ratio", F.col(
        "duration_commercial") / F.col("duration_total")
    )
)

answer.orderBy("commercial_ratio", ascending=False).show(500, False)

""" drop ulla values """
answer_no_null = answer.dropna(how="any",
                               thresh=None,
                               subset=["commercial_ratio"])

answer_no_null.show(500, False)

""" fill null values """
answer_no_null = answer.fillna(
    value=-1,
    subset=["commercial_ratio"])

answer_no_null.show(500, False)
