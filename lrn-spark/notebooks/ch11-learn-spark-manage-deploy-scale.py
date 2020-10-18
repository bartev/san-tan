#!/usr/bin/env python
# coding: utf-8

# # Description

# # Setup

# In[12]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))

get_ipython().run_line_magic('load_ext', 'blackcellmagic')
# in a cell, type %%black

get_ipython().run_line_magic('load_ext', 'autoreload')
# in a cell, type %autoreload


# # Imports

# In[24]:


import os
import os.path as path

import src.data.utils as uts


# # Spark

# In[27]:


from pyspark.sql import SparkSession

# Create a SparkSession
spark = (SparkSession
         .builder
         .master('local[*]')
         .appName("spark-ml-ch-11")
         .config('ui.showConsoleProgress', 'false')
         .getOrCreate())


# # Functions

# See `src.data.utils`

# # Model Management

# * Library versioning
# * Data evolution
# * Order of execution
# * Parallel operations

# # MLflow
# https://mlflow.org

# MLflow is an open source platform that helps developers reproduce and share experiments, manage models, and much more. It provides interfaces in Python, R, and Java/Scala, as well as a REST API. As shown in Figure 11-1, MLflow has four main components:

# * Tracking
#     * Provides APIs to record parameters, metrics, code versions, models, and artifacts such as plots, and text.
# * Projects
#     * A standardized format to package your data science projects and their dependencies to run on other platforms. It helps you manage the model training process.
# * Models
#     * A standardized format to package models to deploy to diverse execution environments. It provides a consistent API for loading and applying models, regardless of the algorithm or library used to build the model.
# * Registry
#     * A repository to keep track of model lineage, model versions, stage transitions, and annotations.
# 

# ## Let’s examine a few things that can be logged to the tracking server:
# 
# * Parameters
#     * Key/value inputs to your code—e.g., hyperparameters like num_trees or max_depth in your random forest
# * Metrics
#     * Numeric values (can update over time)—e.g., RMSE or accuracy values
# * Artifacts
#     * Files, data, and models—e.g., matplotlib images, or Parquet files
# * Metadata
#     * Information about the run, such as the source code that executed the run or the version of the code (e.g., the Git commit hash string for the code version)
# * Models
#     * The model(s) you trained
# 
# 
# By default, the tracking server records everything to the filesystem, but you can specify a database for faster querying, such as for the parameters and metrics. Let’s add MLflow tracking to our random forest code from Chapter 10:

# In[28]:


from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

filePath = db_fname("sf-airbnb/sf-airbnb-clean.parquet")
airbnbDF = spark.read.parquet(filePath)

(trainDF, testDF) = airbnbDF.randomSplit([0.8, 0.2], seed=42)


# In[30]:


import src.models.train_model as tm


# In[32]:


categoricalCols = [field for (field, dataType) in trainDF.dtypes if dataType == 'string']

stages_string_indexer = tm.make_string_indexer_list(categoricalCols)

indexOutputCols = [indexer.getOutputCol() for indexer in stages_string_indexer]


# In[35]:


numericCols = [field for (field, dataType) in trainDF.dtypes
                  if ((dataType == 'double') & (field != 'price'))]


# In[36]:


assemblerInputs = indexOutputCols + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs,
                              outputCol='features')

rf = RandomForestRegressor(labelCol='price', maxBins=40, maxDepth=5, numTrees=100, seed=42)

stages = stages_string_indexer + [vecAssembler, rf]
pipeline = Pipeline(stages=stages)


# In[ ]:




