#!/usr/bin/env python
# coding: utf-8

# # Description - Chapter 6. Spark SQL and Datasets
# 
# In this chapter, we go under the hood to understand Datasets: how to work with Datasets in Java and Scala, how Spark manages memory to accommodate Dataset constructs as part of the unified and high-level API, and costs for using Datasets.

# # Setup

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# In[2]:


from pyspark import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext('local', 'Ch5')

# create a SparkSession
spark = (SparkSession
    .builder
    .appName("ch5 example")
    .getOrCreate())


# This chapter was all about scala and java
# 
# I skipped it

# In[ ]:




