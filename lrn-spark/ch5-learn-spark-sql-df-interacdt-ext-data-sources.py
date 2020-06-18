#!/usr/bin/env python
# coding: utf-8

# # Chapter 5. Spark SQL and DataFrames - Interacting with External Data Sources

# https://learning.oreilly.com/library/view/learning-spark-2nd/9781492050032/ch05.html

# # Setup

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# In[ ]:


from pyspark import SparkContext

sc = SparkContext('local', 'Ch5')


# In[2]:


from pyspark.sql import SparkSession

# create a SparkSession
spark = (SparkSession
    .builder
    .appName("ch4 example")
    .getOrCreate())


# In[3]:


spark


# # Imports

# In[4]:


import os


# # Spark SQL UDFs

# * UDFs operate per session and they will not be persisted in the underlying metastore.

# In[5]:


from pyspark.sql.types import LongType


# In[6]:


# create cubed function
def cubed(s):
    return s * s * s


# In[7]:


# register UDF
# spark.udf.register(name, f, returnType)
spark.udf.register('cubed', cubed, LongType())


# In[9]:


spark.range(1,9).show()


# In[10]:


# generate temporary view
spark.range(1, 9).createOrReplaceTempView('udf_test')


# In[11]:


# Execute cubed function

spark.sql("""
select 
    id,
    cubed(id) as id_cubed
from udf_test
    """).show()


# # pyspark udfs wit pandas udfs

# 3 types of pandas UDFS
# 
# * scalar
# * grouped map
# * grouped aggregate

# ## Scalar pandas UDFs

# In[12]:


import pandas as pd

from pyspark.sql.functions import col, pandas_udf


# In[13]:


from pyspark.sql.types import LongType


# In[14]:


# Declare cubed function

def cubed(a: pd.Series) -> pd.Series:
    return a * a * a


# In[15]:


cubed(pd.Series([1,2,3,4]))


# In[16]:


# Create pandas UDF

cubed_udf = pandas_udf(cubed, returnType=LongType())


# In[17]:


# Create pandas series

x = pd.Series([1, 2, 3])

# function for a pandas_udf executed with a pandas df
print(cubed(x))


# In[18]:


from pyspark.sql.functions import col


# In[19]:


# create a spark data frame

df = spark.range(1, 4)

# execute function as a spark vectorized UDF

df.select('id').show()


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


# In[23]:


df.select('id', cubed_udf(col('id'))).show()


# ### tweaks for apache arrow
# 
# https://spark.apache.org/docs/2.4.5/sql-pyspark-pandas-with-arrow.html

# In[24]:


import numpy as np
import pandas as pd


# In[25]:


# enable Arrow-based columnar data transfers
spark.conf.set('spark.sql.exection.arrow.enabled', 'true')


# In[26]:


# generate a pandas df
p_df = pd.DataFrame(np.random.rand(100, 3))


# In[27]:


# create a spark df from pandas df using arrow
df = spark.createDataFrame(p_df)


# In[28]:


df.show(4)


# In[29]:


# convert the spark df back to a pandas df using arrow
result_pdf = df.select('*').toPandas()
result_pdf


# ### pandas UDFs (a.k.a. vectorized UDFs)
# 
# https://spark.apache.org/docs/2.4.5/sql-pyspark-pandas-with-arrow.html

# In[30]:


import pandas as pd

from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import LongType

# declare the functiom and create UDF
def multiply_func(a, b):
    return a * b

multiply = pandas_udf(multiply_func, returnType=LongType())


# In[31]:


# the function for a pandas udf should be able to execute wit local pandas data
x = pd.Series([1, 2, 3])

print(multiply_func(x, x))


# In[32]:


# create a spark df
df = spark.createDataFrame(pd.DataFrame(x, columns=['x']))


# In[33]:


df.show()


# In[34]:


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

# In[64]:


from pyspark.sql.types import *


# In[74]:


schema = StructType(
    [StructField("celsius", ArrayType(IntegerType()))])

t_list = [[35, 36, 32, 30, 40, 42, 38]], [[31, 32, 34, 55, 56]]

t_list


# In[76]:


t_c = spark.createDataFrame(t_list, schema)

t_c.createOrReplaceTempView("tC")

# Show the DataFrame
t_c.show()


# ## Transform
# 
# ```
# transform(array<T>, function<T, U>): array<U>
# ```

# In[57]:


# Calculate Fahrenheit from Celsius for an array of temperatures

from pyspark.sql.types import *
schema = StructType([StructField("celsius", ArrayType(IntegerType()))])
t_list = [[35, 36, 32, 30, 40, 42, 38]], [[31, 32, 34, 55, 56]]
t_c = spark.createDataFrame(t_list, schema)
t_c.createOrReplaceTempView("tC")
# Show the DataFrame
t_c.show()

# In[78]:


spark.sql("""
SELECT celsius
  FROM tC
""").show()


# In[79]:


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


# In[82]:


# filter temperatures > 38C for array of teperatures

spark.sql("""
select celsius,
filter(celsius, t -> t > 38) as high,
transform(high, t -> ((t * 9) div 5) + 32) as high_f
from tC
""").show()


# ### Multiple steps

# In[84]:


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

# In[88]:


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

# In[93]:


from pyspark.sql.functions import reduce


# In[94]:


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


# In[ ]:




