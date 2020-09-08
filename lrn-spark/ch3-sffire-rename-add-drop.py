#!/usr/bin/env python

"""Select and filter operations on sffire data"""

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName('SFFD').getOrCreate()

fname = 'sffire.parquet'
fire_df = spark.read.parquet(fname)

(
    fire_df.select('Delay')
    .where(col('Delay') > 5)
    .withColumnRenamed('Delay', 'ResponseDelayedMins')
    .show(5, False)
)

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

fire_ts_df.select('IncidentDate', 'OnWatchDate', 'AvailableDtTS').show(5, False)

# drop all at once?
cols = ['CallDate', 'WatchDate', 'AvailableDtTm']
(
    new_fire_df.select(cols)
    .withColumn('IncidentDate', to_timestamp(col('CallDate'), 'MM/dd/yyyy'))
    .withColumn('OnWatchDate', to_timestamp(col('WatchDate'), 'MM/dd/yyyy'))
    .withColumn(
        'AvailableDtTS', to_timestamp(col('AvailableDtTm'), 'MM/dd/yyyy hh:mm:ss a')
    )
    # .drop('CallDate', 'WatchDate', 'AvailableDtTm')
    .drop(*cols)
    .show(5, False)
)

# get distinct years
(
    fire_ts_df.select(year('IncidentDate'), year('OnWatchDate').alias('yrOWD'))
    .distinct()
    .orderBy(year('IncidentDate'))
    .show(5)
)

(
    fire_ts_df.select(year('IncidentDate').alias('incYrDt'))
    .distinct()
    .orderBy('incYrDt')
    .show()
)

(
    fire_ts_df.select(year('IncidentDate'))
    .distinct()
    .orderBy(year('IncidentDate'))
    .show()
)

