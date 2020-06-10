#!/usr/bin/env python
# coding: utf-8

# # Create DataFrame with schema

# In[1]:


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


# In[2]:


# create a SparkSession
spark = (SparkSession
    .builder
    .appName("Example-3_6")
    .getOrCreate())


# Learning Spark, 2nd ed ch 3

# In[3]:


# create a DataFrame using the schema defined above
blogs_df = spark.createDataFrame(data, schema)


# In[4]:


blogs_df.schema


# In[10]:


import pyspark.sql.types as pst


# In[23]:


scm = StructType(
    [StructField('Id', IntegerType(), True),
         StructField('First', StringType(), True),
         StructField('Last', StringType(), True),
         StructField('Url', StringType(), True),
         StructField('Published', StringType(), True),
         StructField('Hits', IntegerType(), True),
         StructField('Campaigns', ArrayType(StringType(), True), True)])


# In[25]:


blogs2_df = spark.createDataFrame(data, scm)
blogs2_df.show()


# In[28]:


blogs2_df.columns


# In[31]:


blogs2_df.selectExpr('Hits * 2 as double','Hits').show()


# In[33]:


import pyspark.sql.functions as F


# In[34]:


blogs2_df.select(F.col('Hits') * 2).show()


# In[36]:


blogs2_df.withColumn('Big Hitters', (F.expr('Hits > 10000'))).show()


# In[44]:


(blogs2_df
 .withColumn('AuthorsId', 
             (F.concat(F.expr('First'),
                       F.expr('Last'), 
                       F.expr('Id'))))
 .select('AuthorsId')
 .show(n=4))


# ## 4 ways to do the same thing

# In[49]:


blogs2_df.select('Hits').show(2)
blogs2_df.select(F.expr('Hits')).show(2)
print('"col" is short for "column"')
blogs2_df.select(F.col('Hits')).show(2)
blogs2_df.select(F.column('Hits')).show(2)


# ## Sort by `Id`

# In[56]:


blogs2_df.sort(F.col('Id').desc()).show()


# Hmm. `$` doesn't work to convert to a column

# In[63]:


blogs_df.sort($'Id').show()


# # Rows

# ## Instantiate a row

# In[64]:


# In Python
from pyspark.sql import Row

blog_row = Row(6, "Reynold", "Xin", "https://tinyurl.6", 255568, 
               "3/2/2015", ["twitter", "LinkedIn"])


# In[67]:


# access using index for individual items

blog_row[1]


# ## Row objects can be used to create DataFrames if you need them for quick interactivity and exploration. 

# In[70]:


# In Python 
from pyspark.sql import Row
from pyspark.sql.types import *


# In[68]:


# using DDL String to define a schema
schema = "`Author` STRING, `State` STRING"
rows = [Row("Matei Zaharia", "CA"), Row("Reynold Xin", "CA")]


# In[71]:


authors_df = spark.createDataFrame(rows, schema)
authors_df.show()


# ## drop a column

# In[215]:


authors_df.drop('State').show()


# # Common DataFrame Operations

# In[76]:


people_file = '/Users/bartev/dev/spark-3.0.0-preview2-bin-hadoop2.7/examples/src/main/resources/people.csv'


# In[74]:


from pyspark.sql.types import *


# ## Programmatic way to define a schema

# In[75]:


people_schema = StructType([StructField('name', StringType(), True),
                           StructField('age', IntegerType(), True),
                           StructField('job', StringType(), True)])


# ## read the file using DataFrameReader using format csv

# In[79]:


people_df = spark.read.csv(people_file, header=True, schema=people_schema, sep=';')


# In[80]:


people_df.show()


# In[92]:


people_tbl = people_df.write.format('parquet').save('people.parquet')


# In[85]:


spark.read.parquet('people.parquet')


# In[93]:


parquet_table = 'people_tbl'
(people_df.write
    .format('parquet')
    .saveAsTable(parquet_table))


# ## Projections and filters

# In[94]:


people_df = spark.read.csv(people_file, header=True, schema=people_schema, sep=';')


# In[97]:


(people_df.select('age')
     .where('age > 30')
    .show())


# In[98]:


movie_fname = '/Users/bartev/dev/github-bv/san-tan/lrn-spark/Data-ML-100k--master/ml-100k/u.item'


# In[105]:


movies_df = spark.read.csv(movie_fname, header=False, sep='|')


# In[132]:


# Rename columns

(movies_df
 .select('_c0', '_c1', '_c2', '_c4')
 .withColumnRenamed('_c0', 'id')
 .withColumnRenamed('_c1', 'title')
 .withColumnRenamed('_c2', 'date')
 .withColumnRenamed('_c4', 'url')
 .where('date > "1996-01-01"')
 .where('id > 30')
#  .schema
 .show(5)
)


# In[127]:


(movies_df
 .select('_c0', '_c1', '_c2', '_c4')
 .withColumnRenamed('_c0', 'id')
 .withColumnRenamed('_c1', 'title')
 .withColumnRenamed('_c2', 'date')
 .withColumnRenamed('_c4', 'url')
 .where('date > "1996-01-01"')
 .where('id > 30')
#  .schema
 .select('date')
 .distinct()
 .count()
)


# In[143]:


# how do I convert column 'id' to an int?

(movies_df
 .select('_c0', '_c1', '_c2', '_c4')
 .withColumnRenamed('_c0', 'id')
 .withColumnRenamed('_c1', 'title')
 .withColumnRenamed('_c2', 'date')
 .withColumnRenamed('_c4', 'url')
#  .where('date > "1996-01-01"')
 .where('id > 30')
 .where(F.col('id') < "38")
#  .schema
 .select('id', 'date')
#  .distinct()
 .show(10, False)
)


# ## Change data types

# In[153]:


movies_df2 = (movies_df
 .select('_c0', '_c1', '_c2', '_c4')
 .withColumnRenamed('_c0', 'id')
 .withColumnRenamed('_c1', 'title')
 .withColumnRenamed('_c2', 'date')
 .withColumnRenamed('_c4', 'url')
 .where('id > 30')
 .select('id', 'date')
)


# In[145]:


movies_df2.count()


# In[146]:


movies_df2.schema


# In[147]:


movies_df2.show(10)


# In[149]:


movies_df2.describe().show()


# ### Date functions

# In[169]:


(movies_df2
 .withColumn('new_date', F.to_date(F.col('date'), 'dd-MMM-yyyy'))
 .withColumn('new_ts', F.to_timestamp(F.col('date'), 'dd-MMM-yyyy'))
 .where(F.col('new_date') < '1990-01-01')
 .show())


# ### order by year

# In[33]:


import pyspark.sql.functions as F


# In[186]:


(movies_df2
 .withColumn('new_date', F.to_date(F.col('date'), 'dd-MMM-yyyy'))
 .withColumn('new_ts', F.to_timestamp(F.col('date'), 'dd-MMM-yyyy'))
 .where(F.col('new_date') < '1990-01-01')
 .orderBy(F.year('new_date'))
 .withColumn('year', F.year('new_date'))
 .withColumn('month', F.month('new_date'))
 .where(F.col('month') != 1)
 .show())


# ## write to csv

# ### With `repartition`

# In[210]:


(movies_df2
    .withColumn('year', F.year(F.to_date(F.col('date'), 'dd-MMM-yyyy')))
     .select('year')
     .distinct()
     .orderBy('year')
     .where('date != "null"')
     .repartition(1)
    .write
 .format('csv')
 .option('header', 'true')
 .save('movie_dates.csv')
)


# ### with `coalesce`

# In[211]:


(movies_df2
    .withColumn('year', F.year(F.to_date(F.col('date'), 'dd-MMM-yyyy')))
     .select('year')
     .distinct()
     .orderBy('year')
     .where('date != "null"')
     .coalesce(1)
    .write
 .format('csv')
 .option('header', 'true')
 .save('movie_dates_coalesce.csv')
)


# ### with `pandas`

# In[213]:


(movies_df2
    .withColumn('year', F.year(F.to_date(F.col('date'), 'dd-MMM-yyyy')))
     .select('year')
     .distinct()
     .orderBy('year')
     .where('date != "null"')
    .toPandas()
 .to_csv('movie_dates_pandas.csv', header=True, index=False)
)


# ## Aggregates

# In[217]:


movies_df = (spark.read
             .csv(movie_fname, header=False, sep='|')
             .select('_c0', '_c1', '_c2', '_c4')
             .withColumnRenamed('_c0', 'id')
             .withColumnRenamed('_c1', 'title')
             .withColumnRenamed('_c2', 'date')
             .withColumnRenamed('_c4', 'url')
             .where('id > 30')
             .select('id', 'date', 'title', 'url')
)
movies_df.show()


# In[240]:


(movies_df
 .where(F.col('date').isNotNull())
 .withColumn('new_date', F.to_date('date', 'dd-MMM-yyyy'))
 .withColumn('year', F.year(F.to_date(F.col('date'), 'dd-MMM-yyyy')))
 .groupBy('year')
 .count()
 .orderBy('count', ascending=False)
 .select(F.sum('count'), F.avg('count'), F.stddev('count'), F.min('count'), F.max('count'))
 .show())


# In[241]:


(movies_df
 .where(F.col('date').isNotNull())
 .withColumn('new_date', F.to_date('date', 'dd-MMM-yyyy'))
 .withColumn('year', F.year(F.to_date(F.col('date'), 'dd-MMM-yyyy')))
 .groupBy('year')
 .count()
 .orderBy('count', ascending=False)
 .select(F.sum('count'), F.avg('count'), F.stddev('count'), F.min('count'), F.max('count'))
 .printSchema())


# In[ ]:




