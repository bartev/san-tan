#!/usr/bin/env python

# read a stream from a local host
from pyspark.sql.functions import explode, split
import pyspark

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('ai.applecare')
logger.setLevel(logging.DEBUG)

logger.debug('setup spark')
spark = (pyspark.sql.SparkSession
         .builder
         .appName('ch2 read stram')
         .getOrCreate())

# spark.sparkContext.setLogLevel('INFO')

df = spark.range(0, 10000, 1, 8)
print(f'>>>>> df.rdd.getNumPartitions(): {df.rdd.getNumPartitions()}')
