# In Python 
from pyspark.sql.types import *
from pyspark.sql import SparkSession
# define schema for our data using DDL 
schema = "`Id` INT,`First` STRING,`Last` STRING,`Url` STRING,`Published` STRING,`Hits` INT,`Campaigns` ARRAY<STRING>"
# create our static data
data = [
    [1, "Jules", "Damji", "https://tinyurl.1", "1/4/2016", 4535, ["twitter", "LinkedIn"]],
    [2, "Brooke","Wenig","https://tinyurl.2", "5/5/2018", 8908, ["twitter", "LinkedIn"]],
    [3, "Denny", "Lee", "https://tinyurl.3","6/7/2019",7659, ["web", "twitter", "FB", "LinkedIn"]],
    [4, "Tathagata", "Das","https://tinyurl.4", "5/12/2018", 10568, ["twitter", "FB"]],
    [5, "Matei","Zaharia", "https://tinyurl.5", "5/14/2014", 40578, ["web", "twitter", "FB", "LinkedIn"]],
    [6, "Reynold", "Xin", "https://tinyurl.6", "3/2/2015", 25568, ["twitter", "LinkedIn"]]
    ]

# main program
if __name__ == "__main__":
    # create a SparkSession
    spark = (SparkSession
        .builder
        .appName("Example-3_6")
        .getOrCreate())
    # create a DataFrame using the schema defined above
    blogs_df = spark.createDataFrame(data, schema)
    # show the DataFrame; it should reflect our table above
    blogs_df.show()
    print()
    # print the schema used by Spark to process the DataFrame
    print(blogs_df.printSchema())

    print()
    print('try again')
    print('NOTE: types are functions')
    print()
    scm = StructType(
        [StructField('Id', IntegerType(), True),
            StructField('First', StringType(), True),
            StructField('Last', StringType(), True),
            StructField('Url', StringType(), True),
            StructField('Published', StringType(), True),
            StructField('Hits', IntegerType(), True),
            StructField('Campaigns', ArrayType(StringType(), True), True)])    
    blogs2_df = spark.createDataFrame(data, scm)
    blogs2_df.show()

    print()
    print(blogs2_df.printSchema())