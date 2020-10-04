#!/usr/bin/env python
# coding: utf-8

# # Description
# ----
# The other version of this notebook did not have the Databricks data.
# This version uses the data provided by Databricks

# # Setup

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# # Imports

# In[3]:


import os
import os.path as path


# ## Setup Spark

# In[2]:


from pyspark.sql import SparkSession

# Create a SparkSession
spark = (SparkSession
  .builder
  .appName("SparkSQLExampleApp")
  .getOrCreate())


# In[11]:


def db_fname(fname):
    import os.path as path
    data_dir = '~/dev/github-bv/LearningSparkV2/databricks-datasets/learning-spark-v2/'
    return path.expanduser(path.join(data_dir, fname))


# In[44]:


csv_file = db_fname('flights/departuredelays.csv')

# Specify schema
schema = """
date STRING,
delay INTEGER,
distance INT,
origin STRING,
destination STRING
"""

df = (spark.read.format('csv')
#     .option('inferSchema', 'true')
      .schema(schema)
      .option('header', 'true')
      .load(csv_file))


# In[45]:


# Create a temporary view
df.createOrReplaceTempView('us_delay_flights_tbl')


# In[46]:


df.printSchema()


# In[47]:


df.show()


# ## Try some sql queries

# In[48]:


q = """
select distance,
    origin,
    destination
from us_delay_flights_tbl
where distance > 1000
order by distance desc
"""
spark.sql(q).show(10)


# In[43]:


q = """
select date,
    delay,
    origin,
    destination
from us_delay_flights_tbl
where delay > 120
and origin = 'SFO' 
and destination = 'ORD'
order by delay desc
"""
spark.sql(q).show(10)


# In[50]:


q = """
select delay,
    origin,
    destination,
    case when delay > 360 then 'very long delays'
        when delay > 120 then 'long delays'
        when delay > 60 then 'short delays'
        when delay > 0 then 'tolerable delays'
        when delay = 0 then 'no delays'
        when delay < 0 then 'early'
    end as flight_delays
from us_delay_flights_tbl
order by origin, delay desc
"""
spark.sql(q).show(10)


# ## Using DataFrame API query

# In[56]:


from pyspark.sql.functions import col, desc, when


# In[52]:


(df.select('distance', 'origin', 'destination')
    .where(col('distance') > 1000)
    .orderBy(desc('distance'))
    .show(10))


# In[54]:


# or, same as above

(df.select('distance', 'origin', 'destination')
    .where(col('distance') > 1000)
    .orderBy('distance', ascending=False)
    .show(10))


# In[59]:


(df.select('delay', 'origin', 'destination',
          when(col('delay') > 360, 'very long delays')
          .when(col('delay') > 120, 'long delays')
          .when(col('delay') > 60, 'short delays')
          .when(col('delay') > 0, 'tolerable delays')
          .when(col('delay') == 0, 'no delays')
          .when(col('delay') < 0, 'early').alias('flight_delays'))
    .orderBy('origin', desc('delay'))
    .show())


# # SQL Tables and Views

# ## Create SQL Databases and Tables

# Default database is called `default`

# In[63]:


spark.sql('create database learn_spark_db')
spark.sql('USE learn_spark_db')


# From here on, any tables will be created in `learn_spark_db`

# ### Create Managed Table

# Fails - needs Hive support

# In[64]:


spark.sql("""create TABLE managed_us_delay_flights_tbl
(date STRING,
delay INT,
distance INT,
origin STRING, 
desitination STRING)""")


# Try using DataFrame API

# In[65]:


csv_file = db_fname('flights/departuredelays.csv')

# Specify schema
schema = """
date STRING,
delay INTEGER,
distance INT,
origin STRING,
destination STRING
"""

flights_df = spark.read.csv(csv_file, schema=schema)


# In[67]:


flights_df.write.saveAsTable('managed_us_delay_flights_tbl')


# ### Create Unmanaged Table

# In[68]:


(flights_df
    .write
    .option('path', '/tmp/data/us_flights_delay')
    .saveAsTable('us_delay_flights_tbl'))


# In[ ]:




