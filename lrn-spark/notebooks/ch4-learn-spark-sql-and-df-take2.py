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


# # Try some sql queries

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


# ## Create Views

# * Views can be global (visible across all `SparkSession`s on a cluster) or session scoped
# 
# * Create and query like a table
# * Views disappear after spark application terminates.
# * tables persist
# ----
# When querying against global view, must use the prefix `global_temp.`

# ### Create using SQL

# In[69]:


q = """
create or replace temp view us_origin_airport_jfk_tmp_view as
select date, delay, origin, destination
from us_delay_flights_tbl
where origin = 'JFK'
"""
spark.sql(q)


# In[70]:


spark.sql("""select * from us_origin_airport_jfk_tmp_view limit 10""").show()


# ### Create using Python and DataFrame API

# In[71]:


df_sfo = spark.sql("""select date, delay, origin, destination
    from us_delay_flights_tbl where origin = 'SFO'""")
df_jfk = spark.sql("""select date, delay, origin, destination
    from us_delay_flights_tbl where origin = 'JFK'""")


# Create temp and global temp view

# In[72]:


df_sfo.createOrReplaceGlobalTempView("us_origin_airport_SFO_global_tmp_view")


# In[73]:


df_jfk.createOrReplaceTempView("us_origin_airport_jfk_tmp_view")


# ### Query View

# In[74]:


q = """select * from global_temp.us_origin_airport_SFO_global_tmp_view"""
spark.sql(q).show()


# In[75]:


q = """select * from us_origin_airport_jfk_tmp_view"""
spark.sql(q).show()


# In[76]:


spark.read.table("us_origin_airport_jfk_tmp_view").show()


# ### Drop view

# In[77]:


spark.sql("drop view if exists us_origin_airport_SFO_global_tmp_view")


# In[79]:


try:
    spark.read.table('us_origin_airport_SFO_global_tmp_view')
except Exception:
    print('failed')


# In[80]:


spark.catalog.dropTempView('us_origin_airport_jfk_tmp_view')


# In[81]:


try:
    spark.read.table('us_origin_airport_jfk_tmp_view')
except Exception:
    print('failed')


# ## View the metadata

# In[82]:


spark.catalog.listDatabases()


# In[90]:


spark.catalog.listTables()


# In[88]:


[(t.database, t.name) for t in spark.catalog.listTables()]


# In[89]:


spark.catalog.listColumns('us_delay_flights_tbl')


# ## Cachine SQL Tables

# In[91]:


spark.sql("cache table us_delay_flights_tbl")


# In[92]:


spark.sql("uncache table us_delay_flights_tbl")


# Lazy cache

# In[93]:


spark.sql("cache lazy table us_delay_flights_tbl")


# In[94]:


spark.sql("uncache table us_delay_flights_tbl")


# ## Read table into DataFrame

# In[95]:


us_flights_df = spark.sql("select * from us_delay_flights_tbl")


# In[96]:


us_flights_df2 = spark.table('us_delay_flights_tbl')


# In[98]:


us_flights_df.show()


# In[97]:


us_flights_df2.show()


# # Data Sources for DataFrames and SQL Tables

# ## `DataFrameReader`
(DataFrameReader
    .format(args)
    .option("key", "value")
    .schema(args)
    .load())
# use `SparkSession.read` to get an instance of a `DataFrameReader`

# In[107]:


file = db_fname('flights/summary-data/parquet/2010-summary.parquet')


# In[108]:


file


# In[110]:


df = spark.read.format('parquet').load(file)


# In[111]:


df.show(3)


# In[112]:


df2 = spark.read.load(file)
df2.show(3)


# In[113]:


file3 = db_fname('flights/summary-data/csv/*')
file3


# In[114]:


df3 = (spark.read.format('csv')
          .option("inferSchema", "true")
          .option('header', 'true')
          .option('mode', 'PERMISSIVE')
          .load(file3))


# In[115]:


df3.show(3)


# In[117]:


file4 = db_fname('flights/summary-data/json/*')
df4 = (spark.read.format('json')
          .load(file4))
df4.show(3)


# ## `DataFrameWriter`
# ----
# To get an instance, use
# 
# `DataFrame.write`
(DataFrameWriter
    .format(args)
    .options(args)
    .bucketBy(args)
    .partitionBy(args)
    .save(path))
    
(DataFrameWriter
    .format(args)
    .options(args)
    .sortBy(args)
    .saveAsTable(table))
# ## Parquet

# ### Reading parquet file into a spark sql table

# In[119]:


parq_fname = db_fname('flights/summary-data/parquet/2010-summary.parquet')


# In[122]:


q = f"""create or replace temporary view us_delay_flights_tbl
using parquet
options(path "{parq_fname}")"""


# In[123]:


print(q)


# In[124]:


spark.sql(q)


# In[126]:


df = spark.sql("select * from us_delay_flights_tbl")
df.show(3)


# ### Write DataFrame to parquet files

# In[127]:


(df.write.format('parquet')
    .mode('overwrite')
    .option('compression', 'snappy')
    .save('/tmp/data/parquet/df_parquet'))


# ### Write DataFrame to spark sql table

# In[131]:


(df.write
    .mode('overwrite')
    .saveAsTable('us_delay_flights_tbl'))


# In[130]:


get_ipython().system('ls -la spark-warehouse/learn_spark_db.db')


# ## JSON

# In[132]:


json_fname = db_fname('flights/summary-data/json/*')


# In[133]:


df_j = spark.read.format('json').load(json_fname)


# In[135]:


df_j.show(3)


# ### Reading json file into a spark sql table

# In[231]:


spark.sql("drop table if exists us_delay_flights_tbl")


# In[235]:


try:
    df_j = spark.table('us_delay_flights_tbl')
    df_j.show()
except Exception:
    print('failed')


# In[233]:


q = f"""create or replace temporary view us_delay_flights_tbl
using json
options(path "{json_fname}")"""
spark.sql(q)


# In[234]:


try:
    df = spark.table('us_delay_flights_tbl')
    df.show(3)
except Exception:
    print('failed')


# In[237]:


df_j.printSchema()


# ### Write DataFrame to JSON file

# In[179]:


(df.limit(10).show())


# In[181]:


(df.limit(10)
    .write
    .format('json')
    .mode('overwrite')
#     .option('compression', 'snappy')
    .save('/tmp/data/json/df_json'))


# In[185]:


(df.limit(10)
    .write
    .format('json')
    .mode('overwrite')
    .option('multiLine', 'true')
#     .option('compression', 'snappy')
    .save('/tmp/data/json/df_json_multi'))


# ## CSV

# In[238]:


csv_fname = db_fname('flights/summary-data/csv/*')


# In[239]:


schema = """
DEST_COUNTRY_NAME STRING, 
ORIGIN_COUNTRY_NAME STRING, 
count INT"""


# In[240]:


df_csv = (spark.read.format('csv')
             .option('header', 'true')
             .schema(schema)
             .option('mode', 'FAILFAST')
             .option('nullValue', '')
             .load(csv_fname))


# In[241]:


df_csv.show(3)


# In[242]:


df_csv.printSchema()


# ### Reading csv file into a spark sql table

# In[243]:


q = f"""
create or replace temporary view us_delay_flights_tbl
using csv
options (
    path "{csv_fname}",
    header "true",
    mode 'FAILFAST',
    nullvalue '',
    inferSchema 'true'
    )
"""
print(q)


# In[244]:


spark.sql("drop table if exists us_delay_flights_tbl")


# In[248]:


try:
    df_c1 = spark.table('us_delay_flights_tbl')
    df_c1.show()
except Exception:
    print('failed')


# In[249]:


spark.sql(q)


# In[250]:


df_c1.printSchema()


# In[251]:


q = f"""
create or replace temporary view us_delay_flights_tbl
using csv
options (
    path "{csv_fname}",
    header "true",
    mode 'FAILFAST',
    nullvalue '',
    schema "{schema}"
    )
"""
print(q)


# In[252]:


spark.sql("drop table if exists us_delay_flights_tbl")


# In[253]:


try:
    df_c = spark.table('us_delay_flights_tbl')
    df_c.show()
except Exception:
    print('failed')


# In[254]:


spark.sql(q)


# In[255]:


try:
    df_c = spark.table('us_delay_flights_tbl')
    df_c.show()
except Exception:
    print('failed')


# In[256]:


df_c.printSchema()


# In[ ]:





# In[ ]:




