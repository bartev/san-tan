#!/usr/bin/env python
# coding: utf-8

# # Chapter 8. Structured Streaming
# 
# Stream processing is defined as a continuous processing of endless streams of data. With the advent of big data, stream processing systems transitioned from single node processing engines to multiple node, distributed processing engines. Traditionally, distributed stream processing has been implemented with a record-at-a-time processing model which looks like this.
# 
# 

# # Setup

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))

from pyspark import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext('local', 'Ch8')

# create a SparkSession
spark = (SparkSession
    .builder
    .appName("ch8 example")
    .getOrCreate())

import os
from pyspark.sql.functions import *

data_dir = '/Users/bartev/dev/gitpie/LearningSparkV2/databricks-datasets/learning-spark-v2/'


# In[ ]:




