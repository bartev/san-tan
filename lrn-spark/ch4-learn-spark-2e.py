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


# In[ ]:




