#!/usr/bin/env python
# coding: utf-8

# # Vectorized User Defined Functions
# 
# Let's compare the performance of UDFs, Vectorized UDFs, and built-in methods.

# In[3]:


from pyspark import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext('local', 'Ch7')

# create a SparkSession
spark = (SparkSession
    .builder
    .appName("ch7 example")
    .getOrCreate())


# Start by generating some dummy data.

# In[7]:


# %python
from pyspark.sql.types import *
from pyspark.sql.functions import col, count, rand, collect_list, explode, struct, count, pandas_udf

df = (spark
      .range(0, 10 * 1000 * 1000)
      .withColumn('id', (col('id') / 1000).cast('integer'))
      .withColumn('v', rand()))

df.cache()
df.count()


# In[5]:


df.show()


# ## Incrementing a column by one
# 
# Let's start off with a simple example of adding one to each value in our DataFrame.

# ### PySpark UDF

# In[6]:


# %python
# @udf("double")
def plus_one(v):
    return v + 1

get_ipython().run_line_magic('timeit', "-n1 -r1 df.withColumn('v', plus_one(df.v)).agg(count(col('v'))).show()")


# Alternate Syntax (can also use in the SQL namespace now)

# In[8]:


# %python
from pyspark.sql.types import DoubleType

def plus_one(v):
    return v + 1
  
spark.udf.register("plus_one_udf", plus_one, DoubleType())

get_ipython().run_line_magic('timeit', '-n1 -r1 df.selectExpr("id", "plus_one_udf(v) as v").agg(count(col(\'v\'))).show()')


# ### Scala UDF
# 
# Yikes! That took awhile to add 1 to each value. Let's see how long this takes with a Scala UDF.

# In[9]:


# %python
df.createOrReplaceTempView("df")


# In[10]:


get_ipython().run_line_magic('scala', '')
import org.apache.spark.sql.functions._

val df = spark.table("df")

def plusOne: (Double => Double) = { v => v+1 }
val plus_one = udf(plusOne)


# In[12]:


get_ipython().run_line_magic('scala', '')
df.withColumn("v", plus_one($"v"))
  .agg(count(col("v")))
  .show()


# Wow! That Scala UDF was a lot faster. However, as of Spark 2.3, there are Vectorized UDFs available in Python to help speed up the computation.
# 
# * [Blog post](https://databricks.com/blog/2017/10/30/introducing-vectorized-udfs-for-pyspark.html)
# * [Documentation](https://spark.apache.org/docs/latest/sql-programming-guide.html#pyspark-usage-guide-for-pandas-with-apache-arrow)
# 
# ![Benchmark](https://databricks.com/wp-content/uploads/2017/10/image1-4.png)
# 
# Vectorized UDFs utilize Apache Arrow to speed up computation. Let's see how that helps improve our processing time.

# [Apache Arrow](https://arrow.apache.org/), is an in-memory columnar data format that is used in Spark to efficiently transfer data between JVM and Python processes. See more [here](https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html).
# 
# Let's take a look at how long it takes to convert a Spark DataFrame to Pandas with and without Apache Arrow enabled.

# In[15]:


get_ipython().run_line_magic('python', '')
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

get_ipython().run_line_magic('timeit', '-n1 -r1 df.toPandas()')


# In[16]:


get_ipython().run_line_magic('python', '')
spark.conf.set("spark.sql.execution.arrow.enabled", "false")

get_ipython().run_line_magic('timeit', '-n1 -r1 df.toPandas()')


# ### Vectorized UDF

# In[18]:


get_ipython().run_line_magic('python', '')
@pandas_udf('double')
def vectorized_plus_one(v):
    return v + 1

get_ipython().run_line_magic('timeit', "-n1 -r1 df.withColumn('v', vectorized_plus_one(df.v)).agg(count(col('v'))).show()")


# Alright! Still not as great as the Scala UDF, but at least it's better than the regular Python UDF!
# 
# Here's some alternate syntax for the Pandas UDF.

# In[20]:


get_ipython().run_line_magic('python', '')
from pyspark.sql.functions import pandas_udf

def vectorized_plus_one(v):
    return v + 1

vectorized_plus_one_udf = pandas_udf(vectorized_plus_one, "double")

get_ipython().run_line_magic('timeit', "-n1 -r1 df.withColumn('v', vectorized_plus_one_udf(df.v)).agg(count(col('v'))).show()")


# ### Built-in Method
# 
# Let's compare the performance of the UDFs with using the built-in method.

# In[22]:


get_ipython().run_line_magic('python', '')
from pyspark.sql.functions import lit

get_ipython().run_line_magic('timeit', "-n1 -r1 df.withColumn('v', df.v + lit(1)).agg(count(col('v'))).show()")


# ## Computing subtract mean
# 
# Up above, we were working with Scalar return types. Now we can use grouped UDFs.

# ### PySpark UDF

# In[25]:


get_ipython().run_line_magic('python', '')
from pyspark.sql import Row
import pandas as pd

@udf(ArrayType(df.schema))
def subtract_mean(rows):
  vs = pd.Series([r.v for r in rows])
  vs = vs - vs.mean()
  return [Row(id=rows[i]['id'], v=float(vs[i])) for i in range(len(rows))]
  
get_ipython().run_line_magic('timeit', "-n1 -r1 (df.groupby('id').agg(collect_list(struct(df['id'], df['v'])).alias('rows')).withColumn('new_rows', subtract_mean(col('rows'))).withColumn('new_row', explode(col('new_rows'))).withColumn('id', col('new_row.id')).withColumn('v', col('new_row.v')).agg(count(col('v'))).show())")


# ### Vectorized UDF

# In[27]:


get_ipython().run_line_magic('python', '')
from pyspark.sql.functions import PandasUDFType

@pandas_udf(df.schema, PandasUDFType.GROUPED_MAP)
def vectorized_subtract_mean(pdf):
	return pdf.assign(v=pdf.v - pdf.v.mean())

get_ipython().run_line_magic('timeit', "-n1 -r1 df.groupby('id').apply(vectorized_subtract_mean).agg(count(col('v'))).show()")

