#!/usr/bin/env python

# Read from json file

import sys
from pyspark.sql import SparkSession

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: python ch3-ex3-7-read-json.py <fname>', file=sys.stderr)
        sys.exit(-1)

    json_fn = sys.argv[1]
    schema = """
        Id INT,
        First STRING,
        Last STRING,
        Url STRING,
        Published STRING,
        Hits INT,
        Campaigns ARRAY<STRING>
        """

    spark = (SparkSession
             .builder
             .appName('Example-3_7')
             .getOrCreate())

    blogs_df = spark.read.schema(schema).json(json_fn)

    # show the DF
    blogs_df.show()

    # print the schema used by Spark
    print('schema')
    print(blogs_df.printSchema())

    print(blogs_df.schema)
