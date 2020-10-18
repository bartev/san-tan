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

spark.sparkContext.setLogLevel('INFO')

port = 8888
logger.debug(f'read Stream from {port}')
lines = (spark
         .readStream
         .format('socket')
         .option('host', 'localhost')
         .option('port', port)
         .load())

# Perform transformations
# Split the lines into words

logger.info('Exploding words')
words = lines.select(
    explode(split(lines.value, " "))
    .alias("word"))

logger.info('word counts')
# Generate running word count
word_counts = words.groupBy('word').count()


logger.info('kafka query')
# write out to the stream to Kafka
query = (word_counts
         .writeStream
         .format('kafka')
         .option('topic', 'output'))


