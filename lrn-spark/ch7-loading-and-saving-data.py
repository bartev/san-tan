#!/usr/bin/env python
# coding: utf-8

# # Description - Chapter 7. Loading and Saving Your Data
# 
# In this chapter, we discuss strategies to organize data such as bucketing and partitioning data for storage, compression schemes, splittable and non-splittable files, and Parquet files.
# 
# Both engineers and data scientists will find parts of this chapter useful, as they evaluate what storage format is best suited for downstream consumption for future Spark jobs using the saved data.

# # Setup

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# In[2]:


from pyspark import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext('local', 'Ch7')

# create a SparkSession
spark = (SparkSession
    .builder
    .appName("ch7 example")
    .getOrCreate())


# In[33]:


import os
from pyspark.sql.functions import *


# In[37]:


data_dir = '/Users/bartev/dev/gitpie/LearningSparkV2/databricks-datasets/learning-spark-v2/'


# # Text Files

# In this chapter, we discuss strategies to organize data such as bucketing and partitioning data for storage, compression schemes, splittable and non-splittable files, and Parquet files.
# 
# Both engineers and data scientists will find parts of this chapter useful, as they evaluate what storage format is best suited for downstream consumption for future Spark jobs using the saved data.
# 
# 

# ## READING TEXT FILES
# 
# Reading is simple. Use an instance of a SparkSession and DataFrameReader to read a text file into a DataFrame in Python or Dataset in Scala.
# 
# 

# In[3]:


lines_df = (spark.read
               .text('./dirty-data.csv'))

lines_df.show(n=10, truncate=False)


# In[6]:


(spark
 .read
 .text('derby.log')
 .show(n=5, truncate=False))


# In[30]:


lines_df = (spark
 .read
 .text('spark-bartev-log.log')
 )

(lines_df
 .filter(col('value').contains('Spark' or 'spark'))
 .show(n=10, truncate=False))


# ## load apache log

# In[47]:


(spark
 .read
 .text(os.path.join(data_dir, 'SPARK_README.md'))
 .filter(col('value').contains('Spark'))
 .show(n=5, truncate=False))


# In[38]:


lines_df = (spark
           .read
           .text(os.path.join(data_dir, 'web1_access_log_20190715-064001.log')))


# In[48]:


lines_df.show(n=10, truncate=False)


# In[49]:


lines_df.first()


# ## How do I parse this log using pyspark?

# ## Writing text files
# 
# Saving text files is just as simple as reading text files. In our first example above, we can save the filtered text README.txt after filtering out all but Spark occurrences in the file.

# In[56]:


lines_readme = (spark
 .read
 .text(os.path.join(data_dir, 'SPARK_README.md'))
)

(lines_readme
 .filter(col('value').contains('Spark'))
 .show(n=30, truncate=False))


# In[52]:


(lines_readme
 .filter(col('value').contains('Spark'))
 .write.text('./storage/SPARK_README.md'))


# In[57]:


spark.read.text('./storage/SPARK_README.md/part-*.txt').show(truncate=False)


# ## Partitioning

# In[60]:


(lines_readme
#  .filter(col('value').contains('Spark'))
 .repartition(4)
 .write
 .format('parquet')
 .mode('overwrite')
 .save('./storage/SPARK_README.parquet')
)


# In[61]:


(spark
 .read
 .parquet('./storage/SPARK_README.parquet/part-*.parquet')
 .show())


# ### Partition by a given field or column

# In[67]:


names_df = (
    spark
    .read
    .csv('./dirty-data.csv', header=True))

names_df.show()


# In[68]:


(names_df
    .write
    .mode('overwrite')
    .format('parquet')
    .partitionBy('name')
    .save('./storage/people.parquet'))


# In[69]:


(names_df
    .write
    .mode('overwrite')
    .format('parquet')
    .partitionBy('age')
    .save('./storage/people.parquet'))


# In[70]:


(names_df
    .write
    .mode('overwrite')
    .format('parquet')
    .partitionBy('age', 'name')
    .save('./storage/people.parquet'))


# ## Bucketing

# ### Create a managed table

# In[73]:


(names_df
 .groupBy('name')
 .count()
 .orderBy(desc('count'))
 .write.format('parquet')
 .bucketBy(3, 'count', 'name')
 .saveAsTable('names_tbl')
)


# ### Examine 'bucketed' properties of the bucketed table

# In[84]:


(spark
 .sql('describe formatted names_tbl')
 .show(truncate=False)
#  .toPandas()
)


# In[83]:


(spark
 .sql('describe names_tbl')
 .show(truncate=False)
#  .toPandas()
)


# In[80]:


peeps_df = (spark.read
 .format('parquet')
 .load('./storage/people.parquet'))

peeps_df.show()


# Not working??

# In[82]:


(peeps_df
    .write
    .format('delta')
    .save('./storage/peeps.delta'))


# In[ ]:





# In[ ]:





# In[ ]:




