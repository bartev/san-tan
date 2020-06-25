#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# # Create an RDD of tuples

# ## setup `SparkContext`

# In[2]:


from pyspark import SparkContext

sc = SparkContext('local', 'First App')

def get_sc():
    """get a SparkContext (don't recreate)"""
    import pyspark

    # don't try to redefine 'sc'
    if not globals().get('sc', False):
        sc = pyspark.SparkContext('local', 'ch3 notebook')
    else:
        print('not redefining sc')
        sc = globals()['sc']
    return scsc = get_sc()
# In[5]:


data_names_ages = [('Brooke', 20),
                   ('Denny', 31),
                   ('Jules', 30),
                   ('TD', 35),
                   ('Brooke', 25)]

dataRDD = sc.parallelize(data_names_ages)


# In[6]:


# Use map and reduceByKey transformations with their 
# lambda expressions to aggregate and then compute average

agesRDD = (dataRDD.map(lambda x, y: (x, (y, 1)))
           .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
           .map(lambda x, y, z: (x, y/z))
          )


# In[8]:


# Same thing, but with a dataframe

from pyspark.sql.functions import avg
from pyspark.sql import SparkSession

# create a SparkSession
spark = (SparkSession
    .builder
    .appName("example")
    .getOrCreate())

data_df = spark.createDataFrame(data_names_ages, ['name', 'age'])

# group names, aggregate age, get avg
avg_df = data_df.groupBy('name').agg(avg('age'))


avg_df.show()


# In[9]:


spark


# # Create DataFrame with schema

# see file `ex3-6-define-schema.py`
# 
# run it using either
# 
# ```
# spark-submit ex3-6-define-schema.py
# ```
# 
# or 
# 
# ```
# python ex3-6-define-schema.py
# ```

# In[10]:


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


# In[11]:


# create a SparkSession
spark = (SparkSession
    .builder
    .appName("Example-3_6")
    .getOrCreate())


# Learning Spark, 2nd ed ch 3

# In[15]:


# create a DataFrame using the schema defined above
blogs_df = spark.createDataFrame(data, schema)


# In[16]:


blogs_df.schema


# In[17]:


# these types are defined in spark.sql.types
scm = StructType(
    [StructField('Id', IntegerType(), True),
         StructField('First', StringType(), True),
         StructField('Last', StringType(), True),
         StructField('Url', StringType(), True),
         StructField('Published', StringType(), True),
         StructField('Hits', IntegerType(), True),
         StructField('Campaigns', ArrayType(StringType(), True), True)])


# In[18]:


blogs2_df = spark.createDataFrame(data, scm)
blogs2_df.show()


# In[55]:


blogs_df.columns


# In[54]:


blogs_df.selectExpr('Hits * 2 as double','Hits').show()


# In[21]:


import pyspark.sql.functions as F


# In[22]:


blogs2_df.select(F.col('Hits') * 2, 'Hits', F.col('Hits') - 1).show()


# In[23]:


blogs2_df.withColumn('Big Hitters', (F.expr('Hits > 10000'))).show()


# In[24]:


(blogs2_df
 .withColumn('AuthorsId', 
             (F.concat(F.expr('First'),
                       F.expr('Last'), 
                       F.expr('Id'))))
 .select('AuthorsId')
 .show(n=4))


# ## Using `expr`

# In[56]:


from pyspark.sql.functions import expr


# ### use `expr` to compute a value

# In[60]:


blogs_df.select(expr('Hits * 2')).show(2)


# ### or use `col` to compute value

# In[61]:


blogs_df.select(col('Hits') * 2).show(2)


# ### Add new columns using `withColumn` and `expr`

# In[65]:


(blogs_df
 .withColumn('Big Hitters', expr('Hits > 10000'))
 .toPandas()
#  .show()
)


# ### use `expr` to concatenate columns

# In[71]:


from pyspark.sql.functions import concat

(blogs_df
 .withColumn('AuthorsId', 
             concat(expr('First'), expr('Last'), expr('Id')))
#  .select(expr('AuthorsId'))
 .select('AuthorsId')
 .show() 
)


# ## 4 ways to do the same thing

# In[52]:


blogs_df.show()


# In[53]:


blogs2_df.show()


# In[73]:


from pyspark.sql.functions import column


# In[74]:


blogs_df.select('Hits').show(2)
blogs_df.select(expr('Hits')).show(2)

print('"col" is short for "column"')
blogs_df.select(col('Hits')).show(2)
blogs_df.select(column('Hits')).show(2)


# ## Sort by `Id`

# In[31]:


from pyspark.sql.functions import col


# In[75]:


blogs_df.sort(col('Id').desc()).show()


# In[79]:


from pyspark.sql.functions import when


# In[87]:


(blogs_df
    .withColumn('new',
               when(blogs_df['Id'].isin([1,3,5]),
                    col('First'))
               .otherwise('*' * 8))
.show())


# NOTE: Ch 3 says that `$` is a spark function that converts something to a column. However, it doesn't work here
blogs_df.sort($'Id'.desc()).show()
# In[42]:


(blogs2_df
 .filter('`Id` < 6')
 .filter('Id > 2')
 .sort(col('Id').desc())
 .show())


# In[48]:


# `where` and `filter` are the same


# In[44]:


# use plain SQL in `filter`
(blogs2_df
 .filter('`Id` < 6')
 .filter('Id > 2')
 .filter("Last like 'Z%'")
 .sort(col('Id').desc())
 .show())


# In[47]:


# use plain SQL in `filter`
(blogs2_df
 .filter('`Id` < 6')
 .filter('Id > 2')
 .where("Last like 'Z%'")
 .sort(col('Id').desc())
 .show())


# # Rows

# ## Instantiate a row

# In[91]:


# In Python
from pyspark.sql import Row

blog_row = Row(6, "Reynold", "Xin", "https://tinyurl.6", 255568, 
               "3/2/2015", ["twitter", "LinkedIn"])


# In[98]:


# access using index for individual items
print(f"index 0: {blog_row[0]}")
print(f"index 1: {blog_row[1]}")

print()

for item in blog_row:
    print(item)


# In[95]:


len(blog_row)


# ## Row objects can be used to create DataFrames if you need them for quick interactivity and exploration. 

# In[106]:


# In Python 
from pyspark.sql import Row
from pyspark.sql.types import *


# In[107]:


# using DDL String to define a schema
schema = "`Author` STRING, `State` STRING"
rows = [Row("Matei Zaharia", "CA"), Row("Reynold Xin", "CA")]


# In[108]:


authors_df = spark.createDataFrame(rows, schema)
authors_df.show()


# In[115]:


schema2 = """
    author string, 
    state string
    """
spark.createDataFrame(rows, schema2).show()


# In[111]:


spark.createDataFrame(rows, ['author', 'state']).show()


# ## drop a column

# In[ ]:


authors_df.drop('State').show()


# # Common DataFrame Operations

# In[117]:


import os


# In[123]:


spark_dir = '/Users/bartev/dev/spark-3.0.0-preview2-bin-hadoop2.7'
people_file_relative = 'examples/src/main/resources/people.csv'
people_file = os.path.join(spark_dir, people_file_relative)


# In[124]:


people_file


# In[125]:


from pyspark.sql.types import *


# ## Programmatic way to define a schema

# In[126]:


people_schema = StructType(
    [StructField('name', StringType(), True),
     StructField('age', IntegerType(), True),
     StructField('job', StringType(), True)])


# ## infer schema from a smaller sample

# In[133]:


(spark
 .read
 .option('samplingRatio', 0.5)
 .option('header', 'true')
 .csv(people_file)
 .show())


# ## read the file using DataFrameReader using format csv

# In[128]:


people_df = spark.read.csv(people_file, 
                           header=True, 
                           schema=people_schema, 
                           sep=';')


# ### Can separate out options

# In[143]:


(spark
 .read
 .option('header', 'true')
 .option('schema', people_schema)
 .option('sep', ';')
 .csv(people_file)
 .show())


# ### option `True` or `"true"` - both ok

# In[146]:


(spark
 .read
 .option('header', True)
 .option('schema', people_schema)
 .option('sep', ';')
 .csv(people_file)
 .show())


# In[129]:


people_df.show()


# ## `DataFrameWriter`

# Parquet
# 
# * default format
# * uses compression
# * preserves schema as part of metadata

# In[149]:


people_tbl = (people_df
              .write
              .format('parquet')
              .mode('overwrite')
              .save('people.parquet'))


# In[151]:


spark.read.parquet('people.parquet').show()


# In[155]:


parquet_table = 'people_tbl'
(people_df.write
    .format('parquet')
    .mode('overwrite')
    .saveAsTable(parquet_table))


# ## Projections and filters

# * projection
#     * `select()`
# * filter
#     * `filter()`
#     * `where()`

# In[156]:


people_df = spark.read.csv(people_file, header=True, 
                           schema=people_schema, sep=';')


# In[157]:


people_df.show()


# ### sql like

# In[162]:


(people_df
 .select('age', 'name')
 .where('age != 30')
 .show())


# ### use `col`

# In[161]:


(people_df
 .select('age', 'name')
 .where(col('age') > 30)
 .show())


# ### use an `expr`

# In[160]:


(people_df
 .select('age', 'name')
 .where(expr('age > 30'))
 .show())


# ## Load movies

# In[165]:


movie_fname = '/Users/bartev/dev/github-bv/san-tan/lrn-spark/Data-ML-100k--master/ml-100k/u.item'


# In[166]:


movies_df = spark.read.csv(movie_fname, header=False, sep='|')


# In[167]:


movies_df.show(3)


# ## Rename columns

# In[169]:


movies = (movies_df
 .select('_c0', '_c1', '_c2', '_c4')
 .withColumnRenamed('_c0', 'id')
 .withColumnRenamed('_c1', 'title')
 .withColumnRenamed('_c2', 'date')
 .withColumnRenamed('_c4', 'url')
)
movies.show(5)


# In[182]:


movies.schema


# In[181]:


# Rename columns

(movies
 .where('date > "1996-01-01"')
 .where('id > 30')
 .show(5)
)


# In[174]:


(movies
 .where('date > "1996-01-01"')
 .where('id > 30')
#  .schema
 .select('date')
 .distinct()
 .count()
)


# In[ ]:


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

# In[183]:


movies_df2 = (movies
 .where('id > 30')
 .select('id', 'date')
)


# In[184]:


movies_df2.count()


# In[185]:


movies_df2.schema


# In[186]:


movies_df2.show(10)


# In[187]:


movies_df2.describe().show()


# ### Date functions

# aside: simple example from help docs

# In[211]:


df = spark.createDataFrame([('2020-06-13 12:52', )], ['t'])

print(df.collect())

print(df.select('t', to_date(df.t).alias('date')).collect())

print(df.select('t', to_date(df.t, 'yyyy-MM-dd').alias('date')).collect())


# ### convert to date, and rename

# In[222]:


df.select(to_date(col('t')).alias('other')).collect()


# In[226]:


(df.select('t', 
          col('t').alias('first'),
          to_date(col('t')).alias('other'))
#  .collect()
#  .show()
 .schema
)


# In[230]:


F.current_timestamp


# In[251]:


from pyspark.sql.functions import to_date, to_timestamp, col, date_format
from pyspark.sql.functions import current_date, current_timestamp

(movies_df2
 .withColumn('new_date', to_date(col('date'), 'dd-MMM-yyyy'))
 .withColumn('new_ts', to_timestamp(col('date'), 'dd-MMM-yyyy'))
 .withColumn('new_ts', date_format(col('new_ts'), 'yyyy.MMM.dd E'))

 .withColumn('cur_date', current_date())
 .withColumn('cur_ts', date_format(current_timestamp(), 'MM-yyyy'))
 .where(col('new_date') < '1990-01-01')
#  .show()
 .printSchema()
#  .schema
)


# ### order by year

# In[ ]:


import pyspark.sql.functions as F


# In[253]:


(movies
 .withColumn('new_date', F.to_date(F.col('date'), 'dd-MMM-yyyy'))
 .withColumn('new_ts', F.to_timestamp(F.col('date'), 'dd-MMM-yyyy'))
 .where(F.col('new_date') < '1990-01-01')
 .orderBy(F.year('new_date'))
 .withColumn('year', F.year('new_date'))
 .withColumn('month', F.month('new_date'))
 .where(F.col('month') != 1)
 .filter('month > 4')
 .show())


# ## write to csv

# ### With `repartition`

# In[256]:


(movies
 .withColumn('year', F.year(F.to_date(F.col('date'), 'dd-MMM-yyyy')))
 .select('year')
 .distinct()
 .orderBy('year')
 .where('date != "null"')
 .repartition(1)
 .write
 .format('csv')
 .option('header', True)
 .mode('overwrite')
 .save('movie_dates.csv')
)


# ### with `coalesce`

# In[264]:


(movies
 .withColumn('year', F.year(F.to_date(F.col('date'), 'dd-MMM-yyyy')))
 .select('year')
 .distinct()
 .orderBy('year')
 .where('date != "null"')
 .coalesce(1)
 .write
 .format('csv')
 .option('header', 'true')
 .mode('overwrite')
 .save('movie_dates_coalesce.csv')
)


# ### with `pandas`

# In[269]:


(movies
 .withColumn('year', F.year(F.to_date(F.col('date'), 'dd-MMM-yyyy')))
 .select('year', 'date')
 .distinct()
 .orderBy('year')
 .where('date != "null"')
 .toPandas()
 .to_csv('movie_dates_pandas.csv', header=True, index=False)
)


# ## Aggregates

# In[270]:


movies.show()


# In[274]:


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

print(f'{movies_df.count()} rows')

movies_df.show()


# In[275]:


movies_df.toPandas().head(3)


# ### String replacement using `regexp_repl`

# In[289]:


from pyspark.sql.functions import split, regexp_replace

(movies_df
 .withColumn('tit2', split('url', 'title-exact\?')[1])
 .withColumn('tit2', regexp_replace('tit2', '%20', ' '))
 .show())


# In[293]:


from pyspark.sql.functions import year


# In[304]:


(movies_df
 .where(col('date').isNotNull())
 .select('date')
 .withColumn('dt', to_date('date', 'dd-MMM-yyyy'))
 .withColumn('yr', year(to_date(col('date'), 'dd-MMM-yyyy')))
 .groupBy('yr')
 .count()
 .orderBy('count', ascending=False)
 .show(truncate=False))


# In[325]:


tmp = (movies_df
 .where(col('date').isNotNull())
 .select('date')
 .withColumn('dt', to_date('date', 'dd-MMM-yyyy'))
 .withColumn('yr', year(to_date(col('date'), 'dd-MMM-yyyy')))
 .groupBy('yr')
 .count()
#  .orderBy('count', ascending=False)
#  .select(F.sum('count'), F.avg('count'), F.stddev('count'), F.min('count'), F.max('count'))
#  .show(truncate=False)
)
tmp.show()


# In[290]:


(movies_df
 .where(F.col('date').isNotNull())
 .withColumn('new_date', F.to_date('date', 'dd-MMM-yyyy'))
 .withColumn('year', F.year(F.to_date(F.col('date'), 'dd-MMM-yyyy')))
 .groupBy('year')
 .count()
 .orderBy('count', ascending=False)
 .select(F.sum('count'), F.avg('count'), F.stddev('count'), F.min('count'), F.max('count'))
 .show())


# In[347]:


(movies_df
 .where(F.col('date').isNotNull())
 .withColumn('new_date', F.to_date('date', 'dd-MMM-yyyy'))
 .withColumn('year', F.year(F.to_date(F.col('date'), 'dd-MMM-yyyy')))
 .groupBy('year')
 .count()
 .orderBy('count', ascending=False)
 .select(F.sum('count'), F.avg('count'), F.stddev('count'), F.min('count'), F.max('count'))
 .explain(True))


# In[ ]:


(movies_df
 .where(F.col('date').isNotNull())
 .withColumn('new_date', F.to_date('date', 'dd-MMM-yyyy'))
 .withColumn('year', F.year(F.to_date(F.col('date'), 'dd-MMM-yyyy')))
 .groupBy('year')
 .count()
 .orderBy('count', ascending=False)
 .select(F.sum('count'), F.avg('count'), F.stddev('count'), F.min('count'), F.max('count'))
 .printSchema())


# # End-to-End dataframe example

# See Databricks Community Edition Notebooks

# # Datasets API

# In[334]:


from pyspark.sql import Row

row = Row(350, True, 'Learning Spark 2E', None)


# In[341]:


[r for r in row]


# In[336]:


type(row)


# In[ ]:




