#!/usr/bin/env python

# Define a schema using the DDL

from pyspark.sql import SparkSession
import pyspark.sql.types as st

# define schema for our data
schema = st.StructType(
    [
        st.StructField("Id", st.IntegerType(), False),
        st.StructField("First", st.StringType(), False),
        st.StructField("Last", st.StringType(), False),
        st.StructField("Url", st.StringType(), False),
        st.StructField("Published", st.StringType(), False),
        st.StructField("Hits", st.IntegerType(), False),
        st.StructField("Campaigns", st.ArrayType(st.StringType()), False),
    ]
)

# create our data
data = [
    [1, "Jules", "Damji", "https://tinyurl.1", "1/4/2016", 4535, ["twitter", "LinkedIn"], ],
    [2, "Brooke", "Wenig", "https://tinyurl.2", "5/5/2018", 8908, ["twitter", "LinkedIn"], ],
    [3, "Denny", "Lee", "https://tinyurl.3", "6/7/2019", 7659, ["web", "twitter", "FB", "LinkedIn"], ],
    [4, "Tathagata", "Das", "https://tinyurl.4", "5/12/2018", 10568, ["twitter", "FB"], ],
    [5, "Matei", "Zaharia", "https://tinyurl.5", "5/14/2014", 40578, ["web", "twitter", "FB", "LinkedIn"], ],
    [6, "Reynold", "Xin", "https://tinyurl.6", "3/2/2015", 25568, ["twitter", "LinkedIn"], ],
]

if __name__ == '__main__':
    spark = SparkSession.builder.appName('Example-3_6').getOrCreate()

    # create a DF using the above schema
    blogs_df = spark.createDataFrame(data, schema)

    # show the DF
    blogs_df.show()

    # print the schema used by Spark
    print('schema')
    print(blogs_df.printSchema())