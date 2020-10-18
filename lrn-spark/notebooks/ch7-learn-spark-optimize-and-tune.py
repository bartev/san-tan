#!/usr/bin/env python
# coding: utf-8

# # Description
# ----
# Besides mitigating costs, we also want to consider how to optimize and tune Spark. In this chapter, we will discuss a set of Spark configurations that enable optimizations, look at Spark’s family of join strategies, and inspect the Spark UI, looking for clues to bad behavior.
# 
# 

# # Setup

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# # Imports

# In[2]:


import os
import os.path as path


# # Spark

# In[66]:


from pyspark.sql import SparkSession

# Create a SparkSession
spark = (SparkSession
         .builder
         .master('local[*]')
         .appName("SparkSQLExampleApp")
         .config('ui.showConsoleProgress', 'false')
         .getOrCreate())


# In[118]:


spark


# # Functions

# In[3]:


def db_fname(fname):
    import os.path as path
    data_dir = '~/dev/github-bv/LearningSparkV2/databricks-datasets/learning-spark-v2/'
    return path.expanduser(path.join(data_dir, fname))


# # View Settings

# In[7]:


get_ipython().system('which pyspark')


# In[8]:


from pyspark.conf import SparkConf
from pyspark.context import SparkContext


# In[79]:


conf = SparkConf()


# In[80]:


conf.getAll()


# In[81]:


spark.sparkContext.master


# In[82]:


sc = spark.sparkContext


# In[90]:


spark.sparkContext.getConf().getAll()


# In[84]:


sc.sparkUser()


# In[25]:


sc.version


# In[26]:


type(spark)


# In[35]:


spark


# ## Print all configs

# In[68]:


def print_configs(session: SparkSession):
    """print all the configs for the current SparkSession
    
    Usage:
    > print_configs(spark)
    
    spark.app.name SparkSQLExampleApp
    spark.rdd.compress True
    spark.driver.port 62816
    spark.serializer.objectStreamReset 100
    spark.app.id local-1602131661660
    spark.master local[*]
    spark.executor.id driver
    spark.submit.deployMode client
    spark.driver.host 192.168.1.17
    spark.ui.showConsoleProgress true
    """        
    conf = spark.sparkContext.getConf()
    conf_dict = {k: v for k, v in conf.getAll()}
    
    for k in sorted(conf_dict.keys()):
        print(k, conf_dict[k])

    
print_configs(spark)


# ## View Spark SQL specific spark configs

# In[103]:


(spark.sql("SET -v")
 .select('key', 'value')
#  .filter("key like '%statistics%'")
 .show(10, truncate=False))


# In[104]:


sc = spark.sparkContext


# In[108]:


spark.sparkContext.getConf().get('spark.dynamicAllocation.enabled')
spark.sparkContext.getConf().set('spark.dynamicAllocation.enabled', "true")


# In[109]:


print_configs(spark)


# In[112]:


spark.sparkContext.getConf().getAll()


# In[116]:


spark.sparkContext.getConf().contains('spark.dynamicAllocation.enabled')


# In[117]:


spark.sparkContext.getConf().set('spark.dynamicAllocation.enabled', "true")


# # Caching and Persistence of data

# Example

# In[121]:


from pyspark.sql.functions import col


# In[148]:


df1 = (spark.range(1 * 10000000)
      .toDF('id')
      .withColumn('square', col('id') * col('id')))


# In[149]:


get_ipython().run_cell_magic('time', '', 'df1.count()')


# In[150]:


get_ipython().run_cell_magic('time', '', 'df1.cache()')


# In[151]:


get_ipython().run_cell_magic('time', '', 'df1.count()')


# In[154]:


get_ipython().run_cell_magic('time', '', 'df.count()')


# In[147]:


df.show()


# In[155]:


df1.createOrReplaceTempView('dfTable')


# In[156]:


spark.sql('cache table dfTable')


# In[157]:


spark.sql('select count(*) from dfTable').show()


# In[ ]:




