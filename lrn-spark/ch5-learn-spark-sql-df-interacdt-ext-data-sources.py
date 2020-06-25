#!/usr/bin/env python
# coding: utf-8

# # Chapter 5. Spark SQL and DataFrames - Interacting with External Data Sources

# https://learning.oreilly.com/library/view/learning-spark-2nd/9781492050032/ch05.html

# # Setup

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# In[2]:


from pyspark import SparkContext

sc = SparkContext('local', 'Ch5')


# In[3]:


from pyspark.sql import SparkSession

# create a SparkSession
spark = (SparkSession
    .builder
    .appName("ch5 example")
    .getOrCreate())


# In[4]:


spark


# In[5]:


sc


# # Imports

# In[6]:


import os


# # Spark SQL UDFs

# * UDFs operate per session and they will not be persisted in the underlying metastore.

# In[7]:


from pyspark.sql.types import LongType


# In[8]:


# create cubed function
def cubed(s):
    return s * s * s


# In[13]:


# register UDF
# spark.udf.register(name, f, returnType)
spark.udf.register('c3', cubed, LongType())


# In[10]:


spark.range(1,9).show()


# In[11]:


# generate temporary view
spark.range(1, 9).createOrReplaceTempView('udf_test')


# In[14]:


# Execute cubed function

spark.sql("""
select 
    id,
    c3(id) as id_cubed
from udf_test
    """).show()


# # pyspark udfs with pandas udfs

# 3 types of pandas UDFS
# 
# * scalar
# * grouped map
# * grouped aggregate

# ## Scalar pandas UDFs

# In[15]:


import pandas as pd

from pyspark.sql.functions import col, pandas_udf


# In[16]:


from pyspark.sql.types import LongType


# In[17]:


# Declare cubed function

def cubed(a: pd.Series) -> pd.Series:
    return a * a * a


# In[19]:


cubed(pd.Series([1,2,3,4]))


# In[20]:


# Create pandas UDF

cubed_udf = pandas_udf(cubed, returnType=LongType())


# In[21]:


# Create pandas series

x = pd.Series([1, 2, 3])

# function for a pandas_udf executed with a pandas df
print(cubed(x))


# In[22]:


cubed(x)


# In[28]:


(pd.DataFrame(data=[1,2,3], columns=['a'])
   .assign(b=lambda x: cubed(x['a'])))


# In[29]:


from pyspark.sql.functions import col


# The following fails due to no `pyarrow`

# In[32]:


# create a spark data frame

df = spark.range(1, 4)

# execute function as a spark vectorized UDF

df.select('id', cubed_udf(col('id'))).show()


# In[33]:


get_ipython().system('python -m pip freeze | grep arrow')


# ## This is failing
# 
# org.apache.arrow errors
# 
# * may only work in spark 3.x
#     * https://spark.apache.org/docs/2.4.5/sql-pyspark-pandas-with-arrow.html
#     
# * the way mentioned above still works
# 
# Solution
# 
# * downgrade pyarrow to < 0.15.0
# * https://stackoverflow.com/questions/58269115/how-to-enable-apache-arrow-in-pyspark/58273294#58273294

# In[22]:


spark.version


# In[40]:


df.select('id', cubed_udf(col('id'))).show()


# ### tweaks for apache arrow
# 
# https://spark.apache.org/docs/2.4.5/sql-pyspark-pandas-with-arrow.html

# In[35]:


import numpy as np
import pandas as pd


# In[36]:


# enable Arrow-based columnar data transfers
spark.conf.set('spark.sql.exection.arrow.enabled', 'true')


# In[37]:


# generate a pandas df
p_df = pd.DataFrame(np.random.rand(100, 3))


# In[38]:


# create a spark df from pandas df using arrow
df = spark.createDataFrame(p_df)


# In[39]:


df.show(4)


# In[41]:


# convert the spark df back to a pandas df using arrow
result_pdf = df.select('*').toPandas()
result_pdf


# ### pandas UDFs (a.k.a. vectorized UDFs)
# 
# https://spark.apache.org/docs/2.4.5/sql-pyspark-pandas-with-arrow.html

# In[42]:


import pandas as pd

from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import LongType

# declare the functiom and create UDF
def multiply_func(a, b):
    return a * b

multiply = pandas_udf(multiply_func, returnType=LongType())


# In[43]:


# the function for a pandas udf should be able to execute wit local pandas data
x = pd.Series([1, 2, 3])

print(multiply_func(x, x))


# In[44]:


# create a spark df
df = spark.createDataFrame(pd.DataFrame(x, columns=['x']))


# In[45]:


df.show()


# In[46]:


# execute the function as a spark vectorized udf
df.select(multiply(col('x'), col('x'))).show()


# ## grouped map (pandas udf)

# https://spark.apache.org/docs/2.4.5/sql-pyspark-pandas-with-arrow.html
# 
# Must define
# 
# * A Python function that defines the computation for each group.
# * A StructType object or a string that defines the schema of the output DataFrame.
# 

# In[36]:


from pyspark.sql.functions import pandas_udf, PandasUDFType

df = spark.createDataFrame(
    [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],
    ("id", "v"))
df.show()


# In[37]:


# function to apply to each group

@pandas_udf("id long, v double", PandasUDFType.GROUPED_MAP)
def subtract_mean(pdf):
    # pdf is a pandas.DataFrame
    v = pdf.v
    return pdf.assign(v=v - v.mean())

df.groupby("id").apply(subtract_mean).show()


# # Higher-Order Functions

# In[50]:


spark


# In[53]:


from pyspark.sql.types import *


# In[56]:


schema = StructType(
    [StructField("celsius", ArrayType(IntegerType()))])

t_list = [[35, 36, 32, 30, 40, 42, 38]], [[31, 32, 34, 55, 56]]

t_list


# In[58]:


t_c = spark.createDataFrame(t_list, schema)

t_c.createOrReplaceTempView("tC")

# Show the DataFrame
t_c.show()


# ## Transform
# 
# ```
# transform(array<T>, function<T, U>): array<U>
# ```

# In[59]:


# Calculate Fahrenheit from Celsius for an array of temperatures


# In[61]:


spark.sql("select * from tc").show()

from pyspark.sql.types import *
schema = StructType([StructField("celsius", ArrayType(IntegerType()))])
t_list = [[35, 36, 32, 30, 40, 42, 38]], [[31, 32, 34, 55, 56]]
t_c = spark.createDataFrame(t_list, schema)
t_c.createOrReplaceTempView("tC")
# Show the DataFrame
t_c.show()

# In[62]:


spark.sql("""
SELECT celsius
  FROM tC
""").show()


# In[65]:


spark.sql("""
SELECT celsius, 
 transform(celsius, t -> ((t * 9) div 5) + 32) as fahrenheit
  FROM tC
""").show()


# ## Filter 

# In[81]:


# filter temperatures > 38C for array of teperatures

spark.sql("""
select celsius,
filter(celsius, t -> t > 38) as high
from tC
""").show()


# In[67]:


# filter temperatures > 38C for array of teperatures

spark.sql("""
select celsius,
filter(celsius, t -> t > 38) as high
from tC
""").show()


# In[68]:


# this won't work

# filter temperatures > 38C for array of teperatures

spark.sql("""
select celsius,
filter(celsius, t -> t > 38) as high,
transform(high, t -> ((t * 9) div 5) + 32) as high_f
from tC
""").show()


# ### Multiple steps

# In[69]:


# filter temperatures > 38C for array of teperatures

spark.sql("""
with step1 as (
    select celsius,
    transform(celsius, t -> ((t * 9) div 5) + 32) as farenheit,
    filter(celsius, t -> t > 38) as high
    from tC)
select *,
    transform(high, t -> ((t * 9) div 5) + 32) as high_f
from step1
""").show()


# ## Exists
# 
# ```
# exists(array<T>, function<T, V, Boolean>): Boolean
# ```
# 
# The exists function returns true if the boolean function holds for any element in the input array.
# 
# 

# In[70]:


# Is there a temperature of 38C in the array of temperatures

spark.sql("""
select 
    celsius,
    exists(celsius, t -> t = 38) as threshold
from tC
""").show()


# ## Reduce
# 
# ```
# reduce(array<T>, B, function<B, T, B>, function<B, R>)
# ```
# 
# The reduce function reduces the elements of the array to a single value by merging the elements into a buffer B using function<B, T, B> and by applying a finishing function<B, R> on the final buffer.
# 
# 

# In[72]:


from pyspark.sql.functions import reduce


# In[73]:


#  Calculate average temperature and convert to F

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


# # Dataframs and Spark SQl common relational operators

# ## Create a `transform` (`pipe`) operator for DataFrames

# In[84]:


import pyspark

from pyspark.sql.dataframe import DataFrame

def transform(self, f):
    return f(self)

DataFrame.transform = transform


# ## Import 2 files and create 2 DFs - airport info and delays

# In[87]:


data_dir = os.path.expanduser('~/dev/data/airline-kaggle')
fn_flights = os.path.join(data_dir, 'flights.csv')
fn_airlines = os.path.join(data_dir, 'airlines.csv')
fn_airports = os.path.join(data_dir, 'airports.csv')


# In[75]:


def lcase_cols(df):
    """return a new DataFrame with all columns lower cased"""
    return df.toDF(*[c.lower() for c in df.columns])


# In[131]:


airlines = (spark
 .read
 .format('csv')
 .option('inferSchema', True)
 .option('header', True)
 .csv(fn_airlines)
 .transform(lcase_cols)
            .withColumnRenamed('iata_code', 'iata')
)

airlines.createOrReplaceTempView('airlines')


# In[132]:


spark.sql('select * from airlines limit 10').show()


# In[124]:


airports = (spark
 .read
 .format('csv')
 .option('inferSchema', True)
 .option('header', True)
 .csv(fn_airports)
 .transform(lcase_cols)
            .withColumnRenamed('iata_code', 'iata')
            .drop('latitude', 'longitude')
)

airports.createOrReplaceTempView('airports')


# In[129]:


spark.sql('select * from airports limit 10').show()


# In[93]:


flights = (spark
 .read
 .format('csv')
 .option('samplingRatio', 0.01)
 .option('inferSchema', True)
 .option('header', True)
 .csv(fn_flights)
 .transform(lcase_cols)
)

flights.createOrReplaceTempView('flights')

spark.sql('select * from flights limit 10').show()
# In[133]:


departure_delays = (
    flights
    .withColumnRenamed('origin_airport', 'origin')
    .withColumnRenamed('destination_airport', 'destination')
    .withColumnRenamed('departure_delay', 'dep_delay')
    .withColumnRenamed('arrival_delay', 'delay')
    .select('year', 'month', 'day', 'delay', 'distance', 
            'origin', 'destination')
)

departure_delays.createOrReplaceTempView('departureDelays')


# In[134]:


spark.sql('select * from departureDelays limit 10').show()


# In[102]:


departure_delays.select('distance', 'delay').dtypes


# In[107]:


spark.sql('select count(*) from departureDelays').show()


# In[109]:


from pyspark.sql.functions import *


# In[179]:


foo = (departure_delays.filter(
    expr("""origin == 'SEA' 
        and destination == 'SFO'
        and delay > 0
        """))
     .select('year', 'month', 'day', 'origin', 'destination', 'delay', 'distance')
)
foo.createOrReplaceTempView('foo')


# In[180]:


spark.sql('select * from foo limit 10').show()


# In[140]:


spark.sql('select count(*) from foo').show()


# ## Unions
# 
# union 2 different DFs with the same schema together

# In[150]:


bar = (departure_delays.filter(
    expr("""origin == 'SFO' 
        and destination == 'SEA'
        and delay > 30
        """))
     .select('year', 'month', 'day', 'origin', 'destination', 'delay')
)
bar.createOrReplaceTempView('bar')


# In[153]:


spark.sql('select count(*) from bar').show()


# In[152]:


spark.sql('select * from bar limit 10').show()


# In[154]:


bar.select('origin').distinct().show()


# In[155]:


# In sql
baz = bar.union(foo)


# In[157]:


baz.groupby('origin').count().show()


# In[158]:


baz.groupby('origin', 'destination').count().show()


# In[166]:


spark.sql("""
select origin, destination, count(*) as count
from bar
group by 1,2""").show()


# ## Join
# 
# By default, a Spark SQL join is an inner join with the options being: inner, cross, outer, full, full_outer, left, left_outer, right, right_outer, left_semi, and left_anti.

# ### in Python

# In[182]:


(foo.join(
    airports,
    airports.iata == foo.origin)
    .select('city', 'state', 'month', 'day', 'year', 'delay', 'distance', 'destination')
    .show())


# ### in SQL

# In[183]:


spark.sql("""
select city,
    state,
    month,
    day,
    year,
    delay,
    distance,
    destination
from foo f
join airports a
on a.iata = f.origin""").show()


# ## windowing

# A window function uses values from the rows in a window to return a set of values (typically in the form of another row). With window functions, it is possible to both operate on a group of rows while still returning a single value for every input row. In this section, we will show how to run a dense_rank window function; there are many other functions as noted in the following table.
# 
# 

# ![image.png](attachment:image.png)

# In[ ]:




