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

# In[2]:


import os
import os.path as path


# # Setup Spark

# In[3]:


from pyspark.sql import SparkSession

# Create a SparkSession
spark = (SparkSession
  .builder
  .appName("SparkSQLExampleApp")
  .getOrCreate())


# In[4]:


def db_fname(fname):
    import os.path as path
    data_dir = '~/dev/github-bv/LearningSparkV2/databricks-datasets/learning-spark-v2/'
    return path.expanduser(path.join(data_dir, fname))


# # User-Defined Functions (UDFs)

# * Operate per session
# * Will not be persisted in the underlying metastore

# In[5]:


from pyspark.sql.types import LongType


# In[6]:


# Create cubed function
def cubed(s):
    return s * s * s

# Register UDF
spark.udf.register("cubed", cubed, LongType())

# Generate temp view
spark.range(1, 9).createOrReplaceTempView("udf_test")


# In[10]:


q = """
select id,
    cubed(id) as id_cubed
from udf_test
"""
spark.sql(q).show()


# ## Evaluation Order and Null Checking in Spark Sql

# 1. Make the UDF `null` aware and do `null` checking in the UDF
# 2. Use `If` or `CASE WHEN` expressions to do the `null` check, and invoke the UDF in a conditional branch

# ## Pandas UDFs

# In[32]:


import pandas as pd

from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import LongType

# Declare the cubed function
def cubed(a: pd.Series) -> pd.Series:
    return a * a * a

# Create the pandas UDF for the cubed function
cubed_udf = pandas_udf(cubed, returnType=LongType())


# ### Usage - Pandas

# In[33]:


x = pd.Series([1, 2, 3])
print(cubed(x))


# In[34]:


(pd.DataFrame(x, columns=['A'])
    .assign(B=lambda z: cubed(z)))


# ### Usage - Spark

# In[35]:


df = spark.range(1, 4)

# Execute function as a Spark vectorized UDF
(df
 .select("id", 
         cubed_udf("id").alias('B'),
         cubed_udf(col("id")).alias('C'))
 .show())


# # Higher-Order Functions in DataFrames and Spark SQL

# ## Option 1: Explode and Collect

# ## Option 2: User-Defined Function

# ## Higher-Order Functions

# Example
-- In SQL
transform(values, value -> lambda expression)
# In[38]:


from pyspark.sql.types import *
schema = StructType([
    StructField('celsius', ArrayType(IntegerType()))
])

t_list = [[35, 36, 32, 30, 40, 42, 38]], [[31, 32, 34, 55, 56]]
t_c = spark.createDataFrame(t_list, schema)
t_c.createOrReplaceTempView('tC')


# In[39]:


t_c.show()


# ### Transform
# 
# ```
# transform(array<T>, function<T, U>): array<U>
# ```

# In[45]:


q = """
select celsius,
transform(celsius, t -> ((t * 9) div 5) + 32) as fahrenheit
from tC
"""
spark.sql(q).show()


# ### Filter
# 
# ```
# filter(array<T>, function<T, Boolean>): array<T>
# ```

# In[46]:


spark.sql("""
SELECT celsius, 
 filter(celsius, t -> t > 38) as high 
  FROM tC
""").show()


# ### Exists
# ```
# exists(array<T>, function<T, V, Boolean>): Boolean
# ```

# In[47]:


# Is there a temperature of 38C in the array of temperatures
spark.sql("""
SELECT celsius, 
       exists(celsius, t -> t = 38) as threshold
  FROM tC
""").show()


# ### Reduce
# ```
# reduce(array<T>, B, function<B, T, B>, function<B, R>)
# ```

# The `reduce()` function reduces the elements of the array to a single value by merging the elements into a buffer `B` using `function<B, T, B>` and applying a finishing `function<B, R>` on the final buffer:

# In[83]:


# Calculate average temperature and convert to F
spark.sql("""
SELECT celsius, 
       reduce(
          celsius, 
          0, 
          (t, acc) -> t + acc, 
          acc -> (acc div size(celsius) * 9 div 5) + 32
        ) as avgFahrenheit 
  FROM tC
""").show()


# # Common Relational Operations
1. Import two files and create two DataFrames, one for airport (airportsna) information and one for US flight delays (departureDelays).

2. Using expr(), convert the delay and distance columns from STRING to INT.

3. Create a smaller table, foo, that we can focus on for our demo examples; it contains only information on three flights originating from Seattle (SEA) to the destination of San Francisco (SFO) for a small time range.


# In[51]:


from pyspark.sql.functions import expr


# In[56]:


tripdelaysFilePath = db_fname('flights/departuredelays.csv')
airportsnaFilePath = db_fname('flights/airport-codes-na.txt')


# In[59]:


airportsna = (spark.read
             .format('csv')
             .options(header='true', 
                     inferSchema='true',
                     sep='\t')
             .load(airportsnaFilePath))

airportsna.createOrReplaceTempView("airports_na")


# In[61]:


departureDelays = (spark.read
                  .format('csv')
                  .options(header='true')
                  .load(tripdelaysFilePath)
                  .withColumn('delay', expr('CAST(delay as INT) as delay'))
                  .withColumn('distance', expr('CAST(distance as INT) as distance')))

departureDelays.createOrReplaceTempView('departureDelaysspark.sql(q)')


# In[62]:


# Create temporary small table
foo = (departureDelays
      .filter(expr("""origin == 'SEA'
                  and destination == 'SFO'
                  and date like '01010%'
                  and delay > 0""")))
foo.createOrReplaceTempView('foo')


# In[63]:


spark.sql('select * from airports_na limit 10').show()


# In[64]:


spark.sql('select * from departureDelays limit 10').show()


# In[65]:


spark.sql('select * from foo').show()


# ## Unions

# In[66]:


bar = departureDelays.union(foo)
bar.createOrReplaceTempView('bar')

bar.filter(expr("""origin == 'SEA'
                  and destination == 'SFO'
                  and date like '01010%'
                  and delay > 0""")).show()


# In[68]:


q = """
select *
from bar
where origin = 'SEA'
and destination = 'SFO'
and date like '01010%'
and delay > 0
"""
spark.sql(q).show()


# ## Joins

# In[71]:


(foo.join(
    airportsna,
    airportsna.IATA == foo.origin)
    .select('City', 'State', 'date', 'delay', 'distance', 'destination')
    .show())


# In[74]:


(foo.join(
    airportsna,
    airportsna['IATA'] == foo['origin'])
    .select('city', 'State', 'DATE', 'delay', 'distance', 'destination')
    .show())


# In[76]:


q = """
select a.city,
    a.state,
    f.date,
    f.delay,
    f.distance,
    f.destination
from foo f
join airports_na a
on a.IATA = f.origin
"""
spark.sql(q).show()


# ## Windowing

# A window function uses values from the rows in a window (a range of input rows) to return a set of values, typically in the form of another row. With window functions, it is possible to operate on a group of rows while still returning a single value for every input row
# Won't work. needs Hive support

q1 = """
DROP TABLE IF EXISTS departureDelaysWindow
"""
q2 = """
CREATE TABLE departureDelaysWindow AS
SELECT origin, destination, SUM(delay) AS TotalDelays 
  FROM departureDelays 
 WHERE origin IN ('SEA', 'SFO', 'JFK') 
   AND destination IN ('SEA', 'SFO', 'JFK', 'DEN', 'ORD', 'LAX', 'ATL') 
 GROUP BY origin, destination
"""
q3 = """
SELECT * FROM departureDelaysWindow;
"""
# spark.sql(q1)
spark.sql(q2)
# spark.sql(q3).show()
# In[91]:


q = """
SELECT origin, destination, SUM(delay) AS TotalDelays 
  FROM departureDelays 
 WHERE origin IN ('SEA', 'SFO', 'JFK') 
   AND destination IN ('SEA', 'SFO', 'JFK', 'DEN', 'ORD', 'LAX', 'ATL') 
 GROUP BY origin, destination
"""
(spark.sql(q)).createOrReplaceTempView('departureDelaysWindow')


# In[92]:


spark.sql('select * from departureDelaysWindow').show()


# In[93]:


q = """
select origin, destination, TotalDelays, rank
from (
    select origin, destination, TotalDelays, dense_rank()
        over (partition by origin order by TotalDelays DESC) as rank
        from departureDelaysWindow
        ) t
where rank <= 3
"""
spark.sql(q).show()


# ## Modifications

# In[94]:


foo.show()


# ### Adding new columns

# In[95]:


from pyspark.sql.functions import expr
foo2 = (foo.withColumn('status',
                      expr("""case when delay <= 10 then 'on-time'
                          else 'delayed' end""")))


# In[97]:


foo2.show()


# ### Drop

# In[98]:


foo3 = foo2.drop('delay')
foo3.show()


# ### Rename columns

# In[99]:


foo4 = foo3.withColumnRenamed('status', 'flight_status')
foo4.show()


# ### Pivoting

# In[100]:


q = """
select destination,
    cast(substring(date, 0, 2) as int) as month,
    delay
from departureDelays
where origin = 'SEA'
"""
spark.sql(q).show()


# In[101]:


q = """
select * from (
    select destination,
        cast(substring(date, 0, 2) as int) as month,
        delay
    from departureDelays
    where origin = 'SEA'
)
pivot (
    cast(avg(delay) as decimal(4, 2)) as AvgDelay,
    max(delay) as MaxDelay
    for month in (1 JAN, 2 feb)
    )
order by destination
"""
spark.sql(q).show()


# ### Summary

# In[ ]:




