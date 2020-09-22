#!/usr/bin/env python

"Import SFFD data"

import pyspark
from pyspark.sql import SparkSession


import os
import sys
os.environ['PYSPARK_PYTHON']=sys.executable

print(f'environ: {os.environ["PYSPARK_PYTHON"]}')


spark = SparkSession.builder.appName('debug-save').getOrCreate()
for item in spark.sparkContext.getConf().getAll():
    print(item)

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

# fname = 'data/sf-fire-calls.csv'
fname = 'data/debug-sf-fire.csv'

fire_df = spark.read.csv(fname, header=True, schema=fire_schema)

fire_df2 = (spark.read
            .option('header', True)
            .schema(fire_schema)
            .csv(fname))

fire_df.show(3)

fire_df.select('ALSUnit', 'NumAlarms').show(3)

fire_df.write.format('parquet').mode('overwrite').save('debug-sffire.parquet')
# fire_df.write.format('parquet').mode('overwrite').saveAsTable('debug_sffire_tbl')
