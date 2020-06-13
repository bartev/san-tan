#!/usr/bin/env python
# coding: utf-8

# # Chapter 4. Spark SQL and DataFrames - Introduction to Built-in Data Sources

# # Setup

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# In[2]:


from pyspark import SparkContext

sc = SparkContext('local', 'Ch4')


# In[3]:


from pyspark.sql import SparkSession

# create a SparkSession
spark = (SparkSession
    .builder
    .appName("ch4 example")
    .getOrCreate())


# In[4]:


spark.version


# # Imports

# In[6]:


import os


# # Load flight data

# This data is from kaggle
# 
# There is more data, and the names are different.
# 
# some of the definitions may be different too

# In[10]:


data_dir = os.path.expanduser('~/dev/data/airline-kaggle')
fn_flights = os.path.join(data_dir, 'flights.csv')
fn_airlines = os.path.join(data_dir, 'airlines.csv')
fn_airports = os.path.join(data_dir, 'airports.csv')


# In[11]:


fn_airlines


# In[51]:


def lcase_cols(df):
    """return a new DataFrame with all columns lower cased"""
    return df.toDF(*[c.lower() for c in df.columns])


# In[52]:


airlines = (spark
 .read
 .format('csv')
 .option('samplingRatio', 0.01)
 .option('inferSchema', True)
 .option('header', True)
 .csv(fn_airlines)
)

airlines = lcase_cols(airlines)


# In[53]:


airlines.show()


# In[54]:


airports = (spark
 .read
 .format('csv')
 .option('inferSchema', True)
 .option('header', True)
 .csv(fn_airports)
)
airports = lcase_cols(airports)


# In[21]:


airports.show()


# In[96]:


flights = (spark
 .read
 .format('csv')
 .option('samplingRatio', 0.01)
 .option('inferSchema', True)
 .option('header', True)
 .csv(fn_flights)
)
flights = lcase_cols(flights)


# In[50]:


flights.count()


# In[97]:


# match columns with example from book
flights = (flights
          .withColumnRenamed('origin_airport', 'origin')
          .withColumnRenamed('destination_airport', 'destination')
           .withColumnRenamed('departure_delay', 'dep_delay')
           .withColumnRenamed('arrival_delay', 'delay')
          )

flights.createOrReplaceTempView('us_delay_flights_tbl')


# In[98]:


flights.printSchema()


# In[77]:


flights.select('airline', 'delay', 'arrival_delay',
              'scheduled_departure', 'departure_time',
              'scheduled_time', 'elapsed_time', 'air_time').show()


# In[68]:


flights.show(3)


# ## SQL Query-1: Find all flights whose distance between origin and destination is greater than 1000 miles in Scala or Python

# In[103]:


spark.sql("""
select distance, origin, destination 
from us_delay_flights_tbl
where distance > 1000
order by distance desc
limit 10
""").show()


# In[129]:


(flights
 .select('distance', 'origin', 'destination')
 .filter('distance > 1000')
 .orderBy('distance', ascending=False)
 .show(10))

# or orderBy(desc('distance'))


# In[131]:


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

# In[104]:


spark.sql("""
select year, month, day, delay, origin, destination
from us_delay_flights_tbl
where 1=1
and delay > 120
and origin = 'SFO'
and destination = 'ORD'
order by delay desc
""").show(10)


# In[132]:


(flights
 .select('year', 'month', 'day', 'delay', 'origin', 'destination')
 .filter('delay > 120')
 .filter('origin = "SFO"')
 .filter('destination = "ORD"')
 .orderBy('delay', ascending=False)
 .show(10))


# In[101]:


spark.sql("""
select year, count(*) as cnt
from us_delay_flights_tbl
group by year
order by cnt
""").show(30)


# In[134]:


flights.groupBy('year').count().show()


# In[137]:


flights.groupBy('month').count().orderBy('month').show()


# ## SQL Query-3: A more complicated query in SQL: let’s label all US flights with a human readable label: Very Long Delay (> 6 hours), Lon g Delay (2 - 6 hours), etc. in a new column called “Flight_Delays” in Scala or Python

# In[114]:


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


# In[142]:


from pyspark.sql.functions import col, when


# In[155]:


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


# In[159]:


spark


# # CREATING SQL DATABASES AND TABLES

# In[161]:


spark.sql("""create database learn_spark_db""")
spark.sql("use learn_spark_db")


# ## create a managed table

# In[163]:


(flights
 .select('year', 'month', 'day',
         'delay', 
         'distance',
         'origin', 
         'destination')
).printSchema()


# In[164]:


# sql

spark.sql("""
create table managed_us_delay_flights_tbl
(year int, month int, day int, delay int, distance int,
origin string, destination string)""")


# In[165]:


# in python DataFrame API

flights.write.saveAsTable('managed_us_delay_flights_tbl')


# ## create an unmanaged table

# In[173]:


# in python DataFrame API

data_dir = '/Users/bartev/dev/github-bv/san-tan/lrn-spark/tmp/data/us_flights_delay'
(flights.select('year', 'month', 'day', 'delay', 'distance', 'origin', 'destination')
    .write
    .option('path', data_dir)
    .saveAsTable('us_delay_flights_tbl'))


# ## create views

# In[168]:


# in python DataFrame API


# In[177]:


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


# In[180]:


df_sfo.createOrReplaceGlobalTempView('us_origin_airport_SFO_global_tmp_view')

df_jfk.createOrReplaceTempView('us_origin_airport_JFK_tmp_view')


# Note: to query the global temp view, use the prefix `global_temp`

# In[182]:


(spark
 .sql("select * from global_temp.us_origin_airport_SFO_global_tmp_view")
 .show())


# In[183]:


(spark.sql("select * from us_origin_airport_JFK_tmp_view").show())


# In[187]:


spark.read.table("us_origin_airport_JFK_tmp_view").show()


# What is the difference between a SparkSession and a Spark application

# In[ ]:




