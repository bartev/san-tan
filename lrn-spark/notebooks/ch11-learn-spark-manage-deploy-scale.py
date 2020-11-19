#!/usr/bin/env python
# coding: utf-8

# # Description
# 
# ----
# https://learning.oreilly.com/library/view/learning-spark-2nd/9781492050032/ch11.html#managingcomma_deployingcomma_and_scaling

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


# ## Start logging with MLflow
# ---
# * start with `mlflow.start_run()`
# * end with `mlflow.end_run()`, or, as in this case, put it in a `with` clause to end automatically

# In[37]:


import mlflow
import mlflow.spark
import pandas as pd


# In[38]:


with mlflow.start_run(run_name='random-forest') as run:
    # Log params: num_trees and max_depth
    mlflow.log_param('num_trees', rf.getNumTrees())
    mlflow.log_param('max_depth', rf.getMaxDepth())
    
    # Log model
    pipelineModel = pipeline.fit(trainDF)
    mlflow.spark.log_model(pipelineModel, 'model')
    
    # Log metrics: RMSE, R2
    predDF = pipelineModel.transform(testDF)
    regressionEvaluator = RegressionEvaluator(predictionCol='prediction',
                                              labelCol='price')
    rmse = regressionEvaluator.setMetricName('rmse').evaluate(predDF)
    r2 = regressionEvaluator.setMetricName('r2').evaluate(predDF)
    mlflow.log_metrics({'rmse': rmse, 'r2': r2})
    
    # Log artifact: feature importance scores
    rfModel = pipelineModel.stages[-1]
    pandasDF = (pd.DataFrame(list(zip(vecAssembler.getInputCols(),
                                      rfModel.featureImportances)),
                            columns=['feature', 'importance'])
               .sort_values(by='importance', ascending=False))
    
    # First write to local filesystem, then tell MLflow wher to find that file
    fname_feat_imp = 'feature_importance.csv'
    pandasDF.to_csv(fname_feat_imp, index=False)
    mlflow.log_artifact(fname_feat_imp)


# ## Seeing the results

# * from the command line, in the directory where the notebook is running, run
# 
# ```
# mlflow ui
# ```
# 
# Then, navigate to `localhost:5000`

# In[43]:


from mlflow.tracking import MlflowClient

client = MlflowClient()
runs = client.search_runs(run.info.experiment_id, 
                          order_by=["attributes.start_time desc"], 
                          max_results=1)

run_id = runs[0].info.run_id
runs[0].data.metrics


# In[44]:


mlflow.run(
  "https://github.com/databricks/LearningSparkV2/#mlflow-project-example", 
  parameters={"max_depth": 5, "num_trees": 100})


# ## BV add cross-validator and run MLflow again

# ### Setup `ParamGrid`

# In[45]:


from pyspark.ml.tuning import ParamGridBuilder
paramGrid = (ParamGridBuilder()
            .addGrid(rf.maxDepth, [2, 4, 6])
            .addGrid(rf.numTrees, [10, 100])
            .build())

from pyspark.ml.tuning import CrossValidator

evaluator = RegressionEvaluator(labelCol='price',
                               predictionCol='prediction',
                               metricName='rmse')
cv = CrossValidator(estimator=rf,
                    evaluator=evaluator,
                    estimatorParamMaps=paramGrid,
                    numFolds=3,
                    parallelism=4,
                    seed=42)

pipeline = Pipeline(stages=stages_string_indexer + [vecAssembler, cv])


# In[68]:


from src.models.utils import parse_param_map


# In[67]:


with mlflow.start_run(run_name='random-forest') as run:
    # Log model
    pipelineModel = pipeline.fit(trainDF)
    mlflow.spark.log_model(pipelineModel, 'model')

    best_model = pipelineModel.stages[-1].bestModel
    
    parameters = parse_param_map(best_model.extractParamMap())
    
    # Log params: num_trees and max_depth
    mlflow.log_param('num_trees', parameters['numTrees'])
    mlflow.log_param('max_depth', parameters['maxDepth'])

    # Log metrics: RMSE, R2
    predDF = pipelineModel.transform(testDF)
    regressionEvaluator = RegressionEvaluator(predictionCol='prediction',
                                              labelCol='price')
    rmse = regressionEvaluator.setMetricName('rmse').evaluate(predDF)
    r2 = regressionEvaluator.setMetricName('r2').evaluate(predDF)
    mlflow.log_metrics({'rmse': rmse, 'r2': r2})
    
    # Log artifact: feature importance scores
#     rfModel = pipelineModel.stages[-1]
    pandasDF = (pd.DataFrame(list(zip(vecAssembler.getInputCols(),
                                      best_model.featureImportances)),
                            columns=['feature', 'importance'])
               .sort_values(by='importance', ascending=False))
    
    # First write to local filesystem, then tell MLflow wher to find that file
    fname_feat_imp = 'feature_importance.csv'
    pandasDF.to_csv(fname_feat_imp, index=False)
    mlflow.log_artifact(fname_feat_imp)


# # Batch

# In[78]:


from mlflow.tracking import MlflowClient

client = MlflowClient()
runs = client.search_runs(run.info.experiment_id, 
                          order_by=["attributes.start_time desc"], 
                          max_results=1)

run_id = runs[0].info.run_id
runs[0].data.metrics


# In[79]:


# Load saved model with MLflow
import mlflow.spark

pipelineModel = mlflow.spark.load_model(f"runs:/{run_id}/model")


# In[80]:


# Generate predictions
inputDF = spark.read.parquet(db_fname('sf-airbnb/sf-airbnb-clean.parquet'))

predDF = pipelineModel.transform(inputDF)


# In[92]:


from pyspark.sql.functions import col, round

(predDF.select('features', round('price', 3).alias('price'), round('prediction', 2).alias('pred'))
 .withColumn('error', round(col('pred') - col('price'), 3))
 .show())


# # Streaming

# * when reading in data, use `spark.readStream()` instead of `spark.read()`

# In[94]:


pipelineModel = mlflow.spark.load_model(f"runs:/{run_id}/model")

# Set up simulated streaming data
repartitionedPath = db_fname('sf-airbnb/sf-airbnb-clean-100p.parquet')
schema = spark.read.parquet(repartitionedPath).schema


# In[95]:


streamingData = (spark
                .readStream
                .schema(schema) # can set the schema this way
                .option('maxFilesPerTrigger', 1)
                .parquet(repartitionedPath))


# In[96]:


# Generate predictions
streamPred = pipelineModel.transform(streamingData)


# In[101]:


streamPred.show()


# # Model Export Patterns for Real-Time Inference
# 
# * MLeap https://mleap-docs.combust.ml
# * ONNX https://onnx.ai

# In[ ]:




