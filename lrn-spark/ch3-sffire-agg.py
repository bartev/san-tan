#!/usr/bin/env python

"""Select and filter operations on sffire data"""

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName('SFFD').getOrCreate()

fname = 'sffire.parquet'
fire_df = spark.read.parquet(fname)

# Converting types
new_fire_df = fire_df.withColumnRenamed("Delay", "ResponseDelayedinMins")
(
    fire_df.select('CallDate')
    .withColumn('IncidentDate', to_timestamp(col('CallDate'), 'MM/dd/yyyy'))
    .printSchema()
)

fire_ts_df = (
    new_fire_df.withColumn('IncidentDate', to_timestamp(col('CallDate'), 'MM/dd/yyyy'))
    .drop('CallDate')
    .withColumn('OnWatchDate', to_timestamp(col('WatchDate'), 'MM/dd/yyyy'))
    .drop('WatchDate')
    .withColumn(
        'AvailableDtTS', to_timestamp(col('AvailableDtTm'), 'MM/dd/yyyy hh:mm:ss a')
    )
    .drop('AvailableDtTm')
)

(
    fire_ts_df.select('CallType')
    .where(col('CallType').isNotNull())
    .groupBy('CallType')
    .count()
    .orderBy('count', ascending=False)
    .show(n=10, truncate=False)
)


import pyspark.sql.functions as F

(
    fire_ts_df.select(
        F.sum('NumAlarms'),
        F.avg('ResponseDelayedinMins'),
        F.min('ResponseDelayedinMins'),
        F.max('ResponseDelayedinMins'),
    ).show()
)

(
    fire_ts_df.select(
        F.sum('NumAlarms').alias('sumAlarms'),
        F.avg('ResponseDelayedinMins').alias('avgDelay'),
        F.min('ResponseDelayedinMins').alias('minDelay'),
        F.max('ResponseDelayedinMins').alias('maxDelay'),
    ).show()
)
