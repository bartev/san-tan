#!/usr/bin/env python
# coding: utf-8

# # Description

# * `spark.ml`
#     * newer API based on DataFrames
# ----    
# * `spark.mllib` (DON'T USE ME)
#     * original ML API based on RDD API
# 

# Data used in this notebook is from SF housing data set from Inside Airbnb.
# 
# `dev/github-bv/LearningSparkV2/databricks-datasets/learning-spark-v2/sf-airbnb`

# # Setup

# In[2]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# # Imports

# In[3]:


import os
import os.path as path


# # Spark

# In[11]:


from pyspark.sql import SparkSession

# Create a SparkSession
spark = (SparkSession
         .builder
         .master('local[*]')
         .appName("spark-ml-ch-10b")
         .config('ui.showConsoleProgress', 'false')
         .getOrCreate())


# # Functions

# In[4]:


def db_fname(fname):
    import os.path as path
    data_dir = '~/dev/github-bv/LearningSparkV2/databricks-datasets/learning-spark-v2/'
    return path.expanduser(path.join(data_dir, fname))


# # Designing ML Pipelines

# * Pipeline API provides a high-level API built on top of DataFrames to organize ML worlflow
# * composed of a series of `transformers` and `estimators`

# ## Definitions

# * Transformer
#     * DF -> DF + 1 or more columns appended
#     * has `.transform()` method
# * Estimator
#     * DF -> Model (Transformer)
#     * learns ('fits') params
#     * has `.fit()` method
# * Pipeline
#     * organize a series of transformers and estimators into a single model
#     * Pipeline is an `estimator`
#     * `pipeline.fit()` returns a `PipelineModel`, which is a `transformer`

# ## Data Ingestion and Exploration

# They've done some cleansing already. See Databricks communitiy edition notebook

# In[8]:


filePath = db_fname('sf-airbnb/sf-airbnb-clean.parquet')


# In[12]:


airbnbDF = spark.read.parquet(filePath)


# In[13]:


airbnbDF.select('neighbourhood_cleansed', 'room_type', 'bedrooms', 'bathrooms', 'number_of_reviews', 'price').show(5)


# In[15]:


trainDF, testDF = airbnbDF.randomSplit([0.8, 0.2], seed=42)
print(f"""There are {trainDF.count()} rows in the training set, and {testDF.count()} in the test set""")


# ## Preparing Features with Transformers

# Linear regression (like many other algorithms in Spark) requires that **all the input features are contained within a single vector in your DataFrame**. Thus, we need to transform our data

# Use `VectorAssembler` to combine all columns into a single vector. https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler

# In[17]:


from pyspark.ml.feature import VectorAssembler
vecAssembler = VectorAssembler(inputCols = ['bedrooms'], outputCol='features')
vecTrainDF = vecAssembler.transform(trainDF)
vecTrainDF.select('bedrooms','features', 'price').show(10)


# In[20]:


vecAssembler.getInputCols()


# In[21]:


vecAssembler.getOutputCol()


# ## Using Estimators to Build Models

# In[26]:


from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol='features', labelCol='price')
lrModel = lr.fit(vecTrainDF)


# In[27]:


type(lrModel)


# In[29]:


type(lr)


# `lr.fit()` returns a `LinearRegressionModel` (lrModel), which is a `transformer`. In other words, the **output** of an estimator’s `fit()` method is a `transformer`. Once the estimator has learned the parameters, the transformer can apply these parameters to new data points to generate predictions

# In[31]:


m = round(lrModel.coefficients[0], 2)
b = round(lrModel.intercept,2)

print(f"""The formula for the linear regression line is  price = {m} x bedrooms + {b}""")


# In[32]:


lrModel.coefficients


# In[36]:


mp = lrModel.extractParamMap()


# In[37]:


mp


# In[42]:


for m in mp:
    print(f'{m.name}: \t{mp[m]}')


# ## Creating a Pipeline

# `pipelineModel` is a `transformer`

# In[44]:


from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[vecAssembler, lr])
pipelineModel = pipeline.fit(trainDF)


# Apply it to our test data set

# In[45]:


preDF = pipelineModel.transform(testDF)


# In[46]:


preDF.columns


# In[47]:


preDF.select('bedrooms', 'features', 'price', 'prediction').show(10)


# ## One-hot encoding

# Convert categorical values into numeric values

# Spark uses a `SparseVector` when the majority of entries are 0, so OHE does not massively increase consumption of memory or compute resources

# Multiple ways to one-hot encode data in Spark
# 
# 1. Use `StringIndexer` and `OneHotEncoder`
# 
#   * apply `StringIndexer` estimator to convert categorical values into category indices (ordered by label frequencies)
#   * pass output to `OneHotEncoder` (`OneHotEncoderEstimator` for us, since we're using Spark 2.4.0)

# In[64]:


from pyspark.ml.feature import OneHotEncoder, StringIndexer, OneHotEncoderEstimator

# categoricalCols = [field for (field, dataType) in trainDF.dtypes if dataType == 'string']
# indexOutputCols = [x + 'Index' for x in categoricalCols]
# oheOutputCols = [x + 'OHE' for x in categoricalCols]

# stringIndexer = StringIndexer(inputCols=categoricalCols,
#                               outputCols=indexOutputCols,
#                               handleInvalid='skip')
# oheEncoder = OneHotEncoder(inputCols=indexOutputCols,
#                            outputCols=oheOutputCols)


# In[65]:


cat_fields = [field for (field, dataType) in trainDF.dtypes if dataType == 'string']


# In[66]:


cat_fields


# In[136]:


def make_string_indexer(col_name):
    """valid values of handleInvalid
    skip (filter rows)
    error (throw an error)
    keep (put in a special additional bucket)
    
    NOTE: spark 3.0 will accept multple columns as input/output
    """
    encoded_col_name = f'{col_name}_Index'
    string_indexer = StringIndexer(inputCol=col_name, 
                                   outputCol=encoded_col_name, 
                                   handleInvalid='keep')
    return string_indexer

def make_one_hot_encoder(col_names):
    """each `*_OHE` column will be a SparseVector after fitting and transformation
    
    Usage:
    ohe_room_type = make_one_hot_encoder(['room_type'])
    encoded_room_type = ohe_room_type.fit(transformed_room_type)

    encoded_room_type.transform(transformed_room_type).show()
    
    +---------------+-----+---------------+-------------+
    |      room_type|price|room_type_Index|room_type_OHE|
    +---------------+-----+---------------+-------------+
    |   Private room|200.0|            1.0|(3,[1],[1.0])|
    |Entire home/apt|250.0|            0.0|(3,[0],[1.0])|
    |Entire home/apt|250.0|            0.0|(3,[0],[1.0])|
    """
    input_col_names = [f'{col_name}_Index' for col_name in col_names]
    output_col_names = [f'{col_name}_OHE' for col_name in col_names]
    estimator = OneHotEncoderEstimator(inputCols=input_col_names,
                                  outputCols=output_col_names)
    return estimator


# In[137]:


stages_cat_str_index = [make_string_indexer(c) for c in cat_fields]


# In[138]:


oheEncoder = make_one_hot_encoder(cat_fields)
oheOutputCols = oheEncoder.getOutputCols()


# In[139]:


oheEncoder.extractParamMap()


# In[140]:


numericCols = [field for (field, dataType) in trainDF.dtypes if ((dataType == 'double') & (field != 'price'))]
numericCols


# In[141]:


assemblerInputs = oheOutputCols + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol='features')


# In[142]:


assemblerInputs


# ### What does `StringIndexer` do?
# 
# * `StringIndexer` creates an estimator that will convert a column of strings to a column of numbers ordered by frequency
# * to use it, `.fit()` your dataframe to get a `model` object out
# * then, `.transform()` new data to append the indexed columns

# #### Make the indexer object (no fitting yet)

# In[92]:


se_1 = make_string_indexer('room_type')


# #### `.fit()` the indexer object (returns a `model`)

# In[81]:


trainDF.select('room_type', 'price').show()


# In[90]:


se_1_model = se_1.fit(trainDF.select('room_type', 'price'))


# #### `.transform()` some data using the fitted indexer object (the `model`)

# In[93]:


transformed_room_type = se_1_model.transform(trainDF.select('room_type', 'price'))
transformed_room_type.show()


# In[102]:


ohe_room_type = make_one_hot_encoder(['room_type'])
encoded_room_type = ohe_room_type.fit(transformed_room_type)

encoded_room_type.transform(transformed_room_type).show()


# In[101]:


transformed_room_type.('room_type').distinct().show()


# ### Back to example

# ## `RFormula`

# In[122]:


from pyspark.ml.feature import RFormula


# In[123]:


rFormula = RFormula(formula='price ~.',
                    featuresCol='features',
                    labelCol='price',
                    handleInvalid='keep')


# In[127]:


rf_transformer = rFormula.fit(trainDF.select('room_type', 'price'))


# In[129]:


rf_transformer.transform(trainDF.select('room_type', 'price')).show()


# In[130]:


rf_transformer = rFormula.fit(trainDF)


# In[135]:


rf_transformer.transform(trainDF).select('room_type', 'price', 'features').show(truncate=False)


# You do not need to one-hot encode categorical features for tree-based methods, and it will often make your tree-based models worse.

# ## Add LinearRegression model

# In[145]:


lr = LinearRegression(labelCol='price', featuresCol='features')
pipeline = Pipeline(stages= stages_cat_str_index + [oheEncoder, vecAssembler, lr])


# In[146]:


pipelineModel = pipeline.fit(trainDF)

predDF = pipelineModel.transform(testDF)

predDF


# In[150]:


predDF.columns


# In[152]:


predDF.select('price', 'prediction', 'features').show()


# # Evaluating Models

# In spark.ml there are classification, regression, clustering, and ranking evaluators (introduced in Spark 3.0).

# ## RMSE (root mean square error)
# 
# use this and R2 since regression

# $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$

# In[154]:


from pyspark.ml.evaluation import RegressionEvaluator
regressionEvaluator = RegressionEvaluator(
    predictionCol='prediction',
    labelCol='price',
    metricName='rmse')


# In[216]:


rmse = regressionEvaluator.setMetricName('rmse').evaluate(predDF)
print(f'RMSE is {rmse:.1f}')


# In[161]:


regressionEvaluator.evaluate(predDF, {regressionEvaluator.metricName: 'rmse'})


# In[162]:


regressionEvaluator.evaluate(predDF, {regressionEvaluator.metricName: 'mae'})


# ### Create a Baseline Model

# In[164]:


from pyspark.sql.functions import avg, lit


# In[166]:


trainDF.select(avg('price')).show()


# In[169]:


trainDF.select(avg('price')).first()


# In[170]:


trainDF.select(avg('price')).first()[0]


# In[171]:


avgPrice = trainDF.select(avg('price')).first()[0]


# Here, we don't need to do a `model.transform(testDF)` to get a prediction.
# 
# Instead, we are assigning the average price as the prediction value.

# In[172]:


predDF_baseline = testDF.withColumn('avgPrediction', lit(avgPrice))


# In[176]:


predDF_baseline.columns


# In[177]:


from pyspark.ml.evaluation import RegressionEvaluator


# In[178]:


regressionMeanEvaluator = RegressionEvaluator(predictionCol='avgPrediction', labelCol='price', metricName='rmse')


# In[181]:


rmse_baseline = regressionMeanEvaluator.evaluate(predDF_baseline)
print(f'RMSE for predicting avg price is: {rmse_baseline:.2f}')


# In[185]:


print(f'RMSE model: {rmse:.2f}\nRMSE baseline: {rmse_baseline:.2f}')
print('The model beat the baseline.')


# ## R2

# * $R^2$ values range from $(-\infty, 1)$

# \begin{align}
# R^2 &= 1 - \frac{SS_{res}}{SS_{tot}} \\
# SS_{tot} &= \sum_{i=1}^n (y_i - \bar{y})^2 \\
# SS_{res} &= \sum_{i=1}^n (y_i - \hat{y})^2
# \end{align}

# In[214]:


r2 = regressionEvaluator.setMetricName('r2').evaluate(predDF)


# In[193]:


regressionEvaluator.evaluate(predDF, {regressionEvaluator.metricName: 'r2'})


# In[189]:


r2


# In[192]:


r2 - (1 - (rmse/rmse_baseline)**2)


# ## Predict price on log scale

# In[200]:


from pyspark.sql.functions import col, log


# In[201]:


logTrainDF = trainDF.withColumn('log_price', log(col('price')))
logTestDF = testDF.withColumn('log_price', log(col('price')))


# In[207]:


log_lr = LinearRegression(labelCol='log_price', featuresCol='features', predictionCol='log_pred')
log_pipeline = Pipeline(stages = stages_cat_str_index + [oheEncoder, vecAssembler, log_lr])


# In[208]:


log_pipeline_model = log_pipeline.fit(logTrainDF)

log_predDF = log_pipeline_model.transform(logTestDF)

log_predDF


# In[209]:


log_predDF.columns


# ### Exponentiate

# In[206]:


from pyspark.sql.functions import col, exp
from pyspark.ml.evaluation import RegressionEvaluator


# In[210]:


log_predDF.withColumn('prediction', exp(col('log_pred'))).select('price', 'prediction', 'log_pred').show()


# In[211]:


expDF = log_predDF.withColumn('prediction', exp(col('log_pred')))


# In[218]:


log_regr_eval = RegressionEvaluator(labelCol='price', predictionCol='prediction')
log_rmse = log_regr_eval.setMetricName('rmse').evaluate(expDF)
log_r2 = log_regr_eval.setMetricName('r2').evaluate(expDF)

print(f'RMSE: {rmse:.2f}')
print(f'r2: {r2:.2f}')
print()
print(f'log RMSE: {log_rmse:.2f}')
print(f'log r2: {log_r2:.2f}')


# Notice: prices are lognormal (the log of the prices is a normal distribution)
# 
# building a model to predict log prices, then exponentiating to get actual price results in a lower RMSE and higher $R^2$

# ## Save the model

# In[219]:


pipelinePath = './lr-pipeline-model'
(pipelineModel
 .write()
 .overwrite()
 .save(pipelinePath))


# ## Load the model

# When loading you need to specify the type of model you are loading (e.g. `LinearRegressionModel` or `LogisticRegressionModel`).
# 
# If you always put transformers/estimators in a `Pipeline`, then you'll always load a `PipelineModel`

# In[220]:


from pyspark.ml import PipelineModel

savedPipelineModel = PipelineModel.load(pipelinePath)


# In[221]:


pred_df_saved = savedPipelineModel.transform(testDF)


# In[225]:


regressionEvaluator.setMetricName('rmse').evaluate(pred_df_saved)


# In[228]:


print(f'rmse: {regressionEvaluator.setMetricName("rmse").evaluate(pred_df_saved)}')
print(f'R2: {regressionEvaluator.setMetricName("r2").evaluate(pred_df_saved)}')


# # Tree Based Models

# In[ ]:




