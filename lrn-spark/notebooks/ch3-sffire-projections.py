#!/usr/bin/env python

"""Select and filter operations on sffire data"""

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName('SFFD').getOrCreate()

fname = 'sffire.parquet'
fire_df = spark.read.parquet(fname)

print('fire_df.schema')
print(fire_df.printSchema())

# Select some rows, and filter on `CallType`
(
    fire_df.select('IncidentNumber', 'AvailableDtTm', 'CallType')
    .where(col('CallType') != 'Medical Incident')
    .show(5, truncate=False)
)


# How many distinct `CallType`s were recorded
(
    fire_df.select('CallType')
    .where(col('CallType').isNotNull())
    .agg(countDistinct('CallType').alias('DistinctCallTypes'))
    .show()
)

# list distinct call types
(
    fire_df.select('CallType')
    .where(col('CallType').isNotNull())
    .distinct()
    .show(n=10, truncate=False)
)

