#!/usr/bin/env python
# coding: utf-8

# # Description

# * Build a random forest model
#     * clean the data
# * serialize it with `mleap`

# # Imports

# In[2]:


# imports for adhoc notebooks
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))

# magics
get_ipython().run_line_magic('load_ext', 'blackcellmagic')
# start cell with `%% black` to format using `black`

get_ipython().run_line_magic('load_ext', 'autoreload')
# start cell with `%autoreload` to reload module


# In[3]:


from mleap import pyspark

from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline

from pyspark.sql import SparkSession

import src.spark.utils as uts


# # Setup spark

# In[4]:


get_ipython().run_line_magic('autoreload', '')

mleap_jars = uts.get_mleap_jars()

spark = (
    SparkSession
    .builder
    .appName('mleap-ex')
    .config('spark.jars', mleap_jars)
    .getOrCreate()
    )


# In[7]:


spark


# In[ ]:




