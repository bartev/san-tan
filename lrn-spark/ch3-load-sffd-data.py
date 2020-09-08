#!/usr/bin/env python

"Import SFFD data"

import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('SFFD').getOrCreate()

fire_schema = """
CallNumber INT,
UnitID STRING,
IncidentNumber INT,
CallType STRING,
CallDate STRING,
WatchDate STRING,
CallFinalDisposition STRING,
AvailableDtTm STRING,
Address STRING,
City STRING,
Zipcode INT,
Battalion STRING,
StationArea STRING,
Box STRING,
OriginalPriority STRING,
Priority STRING,
FinalPriority INT,
ALSUnit BOOLEAN,
CallTypeGroup STRING,
NumAlarms INT,
UnitType STRING,
UnitSequenceInCallDispatch INT,
FirePreventionDistrict STRING,
SupervisorDistrict STRING,
Neighborhood STRING,
Location STRING,
RowID STRING,
Delay FLOAT
"""

fname = 'data/sf-fire-calls.csv'

fire_df = spark.read.csv(fname, header=True, schema=fire_schema)

fire_df2 = (spark.read
            .option('header', True)
            .schema(fire_schema)
            .csv(fname))

fire_df.show(3)

fire_df.select('ALSUnit', 'NumAlarms').show(3)

fire_df.write.format('parquet').save('sffire.parquet')
fire_df.write.format('parquet').saveAsTable('sffire_tbl')
