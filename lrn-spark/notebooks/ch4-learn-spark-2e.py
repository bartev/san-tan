#!/usr/bin/env python
# coding: utf-8

# # Chapter 4. Spark SQL and DataFrames - Introduction to Built-in Data Sources

# # Setup

# In[1]:





# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# In[ ]:


from pyspark import SparkContext

sc = SparkContext('local', 'Ch4')


# In[2]:


from pyspark.sql import SparkSession

# create a SparkSession
spark = (SparkSession
    .builder
    .appName("ch4 example")
    .getOrCreate())


# In[3]:


spark.version


# # Imports

# In[5]:


import os


# # Load flight data

# This data is from kaggle
# 
# There is more data, and the names are different.
# 
# some of the definitions may be different too

# In[6]:


data_dir = os.path.expanduser('~/dev/data/airline-kaggle')
fn_flights = os.path.join(data_dir, 'flights.csv')
fn_airlines = os.path.join(data_dir, 'airlines.csv')
fn_airports = os.path.join(data_dir, 'airports.csv')


# In[32]:


fn_airlines


# In[7]:


def lcase_cols(df):
    """return a new DataFrame with all columns lower cased"""
    return df.toDF(*[c.lower() for c in df.columns])


# In[8]:


airlines = (spark
 .read
 .format('csv')
 .option('inferSchema', True)
 .option('header', True)
 .csv(fn_airlines)
)
airlines = lcase_cols(airlines)


# In[9]:


airlines.show()


# In[42]:


airports = (spark
 .read
 .format('csv')
 .option('inferSchema', True)
 .option('header', True)
 .csv(fn_airports)
)
airports = lcase_cols(airports)


# In[43]:


airports.show()


# In[44]:


flights = (spark
 .read
 .format('csv')
 .option('samplingRatio', 0.01)
 .option('inferSchema', True)
 .option('header', True)
 .csv(fn_flights)
)
flights = lcase_cols(flights)


# In[45]:


flights.count()


# In[46]:


# match columns with example from book
flights = (flights
          .withColumnRenamed('origin_airport', 'origin')
          .withColumnRenamed('destination_airport', 'destination')
           .withColumnRenamed('departure_delay', 'dep_delay')
           .withColumnRenamed('arrival_delay', 'delay')
          )

flights.createOrReplaceTempView('us_delay_flights_tbl')


# In[47]:


flights.printSchema()


# In[49]:


flights.select('airline', 'delay', 'delay',
              'scheduled_departure', 'departure_time',
              'scheduled_time', 'elapsed_time', 'air_time').show()


# In[50]:


flights.show(3)


# ## SQL Query-1: Find all flights whose distance between origin and destination is greater than 1000 miles in Scala or Python

# In[51]:


spark.sql("""
select distance, origin, destination 
from us_delay_flights_tbl
where distance > 1000
order by distance desc
limit 10
""").show()


# In[52]:


(flights
 .select('distance', 'origin', 'destination')
 .filter('distance > 1000')
 .orderBy('distance', ascending=False)
 .show(10))

# or orderBy(desc('distance'))


# In[53]:


from pyspark.sql.functions import desc

(flights
 .select('distance', 'origin', 'destination')
 .filter('distance > 1000')
 .orderBy(desc('distance'))
 .show(10))

# or orderBy(desc('distance'))


# ## SQL Query-2: Find all flights with at least 2 hour delays between San Francisco (SFO) and Chicago (ORD) in Scala or Python
# 
# 

# In[54]:


spark.sql("""
select year, month, day, delay, origin, destination
from us_delay_flights_tbl
where 1=1
and delay > 120
and origin = 'SFO'
and destination = 'ORD'
order by delay desc
""").show(10)


# In[55]:


(flights
 .select('year', 'month', 'day', 'delay', 'origin', 'destination')
 .filter('delay > 120')
 .filter('origin = "SFO"')
 .filter('destination = "ORD"')
 .orderBy('delay', ascending=False)
 .show(10))


# In[56]:


spark.sql("""
select year, count(*) as cnt
from us_delay_flights_tbl
group by year
order by cnt
""").show(30)


# In[57]:


flights.groupBy('year').count().show()


# In[58]:


flights.groupBy('month').count().orderBy('month').show()


# ## SQL Query-3: A more complicated query in SQL: let’s label all US flights with a human readable label: Very Long Delay (> 6 hours), Lon g Delay (2 - 6 hours), etc. in a new column called “Flight_Delays” in Scala or Python

# In[59]:


spark.sql("""
select delay,
    origin,
    destination,
    case when delay > 360 then 'very long'
         when delay > 120 then 'long'
         when delay > 60 then 'short'
         when delay > 0 then 'tolerable'
         when delay = 0 then 'no delay'
         else 'early'
    end as flight_delays
from us_delay_flights_tbl
order by delay desc
""").show(20)


# In[60]:


from pyspark.sql.functions import col, when


# In[61]:


(flights
 .select('delay', 
         'origin', 
         'destination',
         when(col('delay') > 360, 'very long')
         .when(col('delay') > 120, 'long')
         .when(col('delay') > 60, 'short')
         .when(col('delay') > 0, 'tolerable')
         .when(col('delay') == 0, 'no delay')
         .when(col('delay') < 0, 'early')
         .otherwise('unknown')
         .alias('flight_delays'))
 .orderBy(desc('delay'))
#  .groupBy('flight_delays')
#  .count()
.show())


# In[62]:


spark


# # CREATING SQL DATABASES AND TABLES

# In[63]:


spark.sql("""create database learn_spark_db""")
spark.sql("use learn_spark_db")


# ## create a managed table

# In[64]:


(flights
 .select('year', 'month', 'day',
         'delay', 
         'distance',
         'origin', 
         'destination')
).printSchema()


# In[65]:


# sql

spark.sql("""
create table managed_us_delay_flights_tbl
(year int, month int, day int, delay int, distance int,
origin string, destination string)""")


# In[66]:


# in python DataFrame API

flights.write.saveAsTable('managed_us_delay_flights_tbl')


# ## create an unmanaged table

# In[67]:


# in python DataFrame API

data_dir = '/Users/bartev/dev/github-bv/san-tan/lrn-spark/tmp/data/us_flights_delay'
(flights.select('year', 'month', 'day', 'delay', 'distance', 'origin', 'destination')
    .write
    .option('path', data_dir)
    .saveAsTable('us_delay_flights_tbl'))


# ## create views

# Note: 
# 
# Views can be
# 
# * global - visible across all SparkSessions on a given cluster
# * temporary - visible only to a single SparkSession
# 
# Views disappear after your Spark application or notebook terminates
# 
# to query the global temp view, use the prefix `global_temp`

# In[68]:


# in python DataFrame API


# In[69]:


df_sfo = spark.sql("""
select year,
    month,
    day,
    delay,
    distance,
    origin,
    destination 
from us_delay_flights_tbl 
where origin = 'SFO'""")

df_jfk = spark.sql("""
select year,
    month,
    day,
    delay,
    distance,
    origin,
    destination 
from us_delay_flights_tbl 
where origin = 'JFK'""")


# In[70]:


df_sfo.createOrReplaceGlobalTempView('us_origin_airport_SFO_global_tmp_view')

df_jfk.createOrReplaceTempView('us_origin_airport_JFK_tmp_view')


# In[71]:


(spark
 .sql("select * from global_temp.us_origin_airport_SFO_global_tmp_view")
 .show())


# In[72]:


spark.sql("select * from us_origin_airport_JFK_tmp_view").show()


# In[73]:


spark.read.table("us_origin_airport_JFK_tmp_view").show()


# ### Question
# What is the difference between a SparkSession and a Spark application

# ## Drop view

# ### drop using sql

# In[74]:


spark.sql("select count(*) as jfk from us_origin_airport_JFK_tmp_view").show()
spark.sql("select count(*) as sfo from global_temp.us_origin_airport_SFO_global_tmp_view").show()


# In[75]:


spark.sql("""drop view if exists us_origin_airport_SFO_global_tmp_view""")

spark.sql("""drop view if exists us_origin_airport_JFK_tmp_view""")


# ### create again

# In[76]:


df_sfo.createOrReplaceGlobalTempView('us_origin_airport_SFO_global_tmp_view')
df_jfk.createOrReplaceTempView('us_origin_airport_JFK_tmp_view')


# ### drop using `catalog`

# In[77]:


spark.catalog.dropGlobalTempView('us_origin_airport_SFO_global_tmp_view')
spark.catalog.dropTempView('us_origin_airport_JFK_tmp_view')


# ## Look at Metadata via `catalog`

# ### `listDatabases`

# In[78]:


dbs = spark.catalog.listDatabases()


# In[79]:


[print(f'{db}\n') for db in dbs]


# In[80]:


len(dbs)


# In[81]:


dbs[0]


# In[82]:


dbs[1]


# ### `listTables`

# In[83]:


tbls = spark.catalog.listTables()


# In[84]:


[print(f'{tbl}\n') for tbl in tbls]


# ### `listColumns`

# In[85]:


spark.catalog.listColumns('us_delay_flights_tbl')


# ## Cache sql tables

# In[86]:


spark.sql("cache table us_delay_flights_tbl")


# In[87]:


spark.sql("uncache table us_delay_flights_tbl")


# # Read table into dataframe

# In[88]:


us_flights_df = spark.sql('select * from us_delay_flights_tbl')


# In[89]:


# No need for `read` here

us_flights_df2 = spark.table('us_delay_flights_tbl')


# In[90]:


us_flights_df.printSchema()


# # Data Sources for DataFrames and SQL Tables

# ## DataFrameReader
# 
# * access from a `spark` session
# 
# `DataFrameReader.format(args).option("key", "value").schema(args).load()`

# ## DataFrameWriter
# 
# * access from a `DataFrame`

# ```
# DataFrameWriter.format(args).option(args).bucketBy(args).partitionBy(args).save()
# DataFrameWriter.format(args).option(args).sortBy(args).saveAsTable()
# DataFrame.write or DataFrame.writeStream
# ```

# Options
# 
# * format: parque (default), csv, txt, ...
# * mode: append, overwrite, ignore, error, errorifexists (default)
# * save: /path/to/data/source (can be empty if specified path in options)
# * saveAsTable: table_name

# ## Parquet

# ### read DataFrame from parquet

# In[91]:


file = 'people.parquet'
spark.read.format('parquet').load(file).show()


# In[92]:


spark.read.load(file).show()


# ### Read parquet file into a spark sql table

# In[93]:


q = """
create or replace temporary view peeps_tbl
using parquet
options(
    path 'people.parquet')
    """


# In[94]:


spark.sql(q)


# then read into a DataFrame

# In[95]:


spark.sql('select * from peeps_tbl').show()


# ### write DataFrame to parquet

# In[96]:


from pyspark.sql.functions import col, lit


# In[97]:


df = (spark.sql('select * from peeps_tbl')
     .withColumn('foo', lit('red')))

df.show()


# #### overwrite

# In[98]:


(df.write
    .mode('overwrite')
    .option('compression', 'snappy')
    .save('/tmp/data/parquet/df_peeps_foo'))


# In[99]:


spark.read.load('/tmp/data/parquet/df_peeps_foo/').show()


# #### overwrite with a new column

# In[100]:


(df.withColumn('bar', lit('blue'))
    .write
    .mode('overwrite')
    .option('compression', 'snappy')
    .save('/tmp/data/parquet/df_peeps_foo/'))


# In[101]:


spark.read.load('/tmp/data/parquet/df_peeps_foo/').show()


# #### append

# In[102]:


(df.withColumn('bar', lit('orange'))
    .write
    .mode('append')
    .option('compression', 'snappy')
    .save('/tmp/data/parquet/df_peeps_foo/'))


# In[103]:


spark.read.load('/tmp/data/parquet/df_peeps_foo/').show()


# #### Add a column (won't show up in table)

# In[104]:


(df.withColumn('baz', lit('green'))
    .write
    .mode('append')
    .option('compression', 'snappy')
    .save('/tmp/data/parquet/df_peeps_foo/'))


# In[105]:


spark.read.load('/tmp/data/parquet/df_peeps_foo/').show()


# In[106]:


(df
    .write
    .mode('append')
    .option('compression', 'snappy')
    .save('/tmp/data/parquet/df_peeps_foo/'))


# In[107]:


spark.read.load('/tmp/data/parquet/df_peeps_foo/').show()


# ## json

# In[108]:


import numpy as np
import pandas as pd


# In[109]:


from numpy.random  import MT19937
from numpy.random import RandomState, SeedSequence


# ### Create random df and write to json

# #### aside: numpy `RandomState` - how does this work?

# In[118]:


rs = RandomState(MT19937(SeedSequence(1)))

np.random.randint(0, 15, size=(5,3))


# #### using `np.random.seed`

# In[125]:


np.random.seed(1)

# np.random.randint(0, 15, size=(5,3))

data = np.random.randint(0, 5*3, size=(5, 3))
columns = ['a', 'b', 'c']
toy_df = pd.DataFrame(data, columns=columns)
toy_df


# In[160]:


file = 'toy_df2.json'
print(toy_df.to_json(orient='records', indent=2))


# ### read json file into DataFrame

# If the json has records on separate lines (as when I save with `indent=2`), then set `multiLine = True`

# In[170]:


# this is more human readable

toy_df.to_json(file, orient='records', indent=2)


# In[164]:


# Fail
spark.read.format('json').option('multiLine', False).load(file).show()


# In[165]:


# Succeed
spark.read.format('json').option('multiLine', True).load(file).show()


# In[171]:


# less readable

toy_df.to_json(file, orient='records', indent=0)


# In[172]:


# Succeed
spark.read.format('json').option('multiLine', True).load(file).show()


# In[173]:


# Succeed
spark.read.format('json').option('multiLine', False).load(file).show()


# ### read json file into spark sql table

# In[174]:


spark.sql(f"""
create or replace temporary view toy_tbl
using json
options (
path = '{file}')
""")


# In[175]:


spark.sql('select * from toy_tbl').show()


# ### write DataFrame to json file

# In[179]:


toy = spark.read.format('json').option('multiLine', True).load(file)


# In[181]:


toy.show()


# In[189]:


(toy.write.format('json')
    .mode('overwrite')
    .save('/tmp/data/json/toy.json'))


# In[190]:


spark.stop()


# In[ ]:




