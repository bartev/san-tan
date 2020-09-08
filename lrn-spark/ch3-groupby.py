#!/usr/bin/env python

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg


# Create a DF using SparkSession
spark = SparkSession.builder.appName("AuthorsAges").getOrCreate()


data_df = spark.createDataFrame(
    [("Brooke", 20), ("Denny", 31), ("Jules", 30), ("TD", 35), ("Brooke", 25)],
    ["name", "age"],
)

# group same names together
# aggregate ages
# compute avg

avg_df = data_df.groupBy("name").agg(avg("age"))

avg_df.show()
