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
# * Model
#     * a Transformer which will return predictions

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
# 
# https://community.cloud.databricks.com/?o=6130133108755320#notebook/3114109954854034/command/3114109954854035

# Decision trees:
# 
# * scale-invariant
# * depth = longest path from root node to any leaf node
#     * if depth is too large, risk overfitting
#     * if too small, underfit
# * feature prep
#     * no need to standardize (scale) data
#     * take care about how to prepare categorical features
#     * pass to `StringIndexer`
#     
# **Do not OHE variables**

# In[229]:


from pyspark.ml.regression import DecisionTreeRegressor


# In[230]:


dt = DecisionTreeRegressor(labelCol='price')


# In[231]:


# filter for just numeric columns and exclude price (the label)
numericCols = [field for (field, dataType) in trainDF.dtypes
               if ((dataType == 'double') & (field != 'price'))]


# Since we're on Spark 2.4, we `StringIndexer` only takes a single column.
# 
# See this function that was defined above.

# In[244]:


categoricalCols = [field for (field, dataType) in trainDF.dtypes if dataType == 'string']


# In[237]:


# indexOutputCols = [x + 'Index' for x in categoricalCols]
# oheOutputCols = [x + 'OHE' for x in categoricalCols]

# stringIndexer = StringIndexer(inputCols=categoricalCols,
#                               outputCols=indexOutputCols,
#                               handleInvalid='skip')
# oheEncoder = OneHotEncoder(inputCols=indexOutputCols,
#                            outputCols=oheOutputCols)


# In[238]:


categoricalCols


# In[233]:


# def make_string_indexer(col_name):
#     """valid values of handleInvalid
#     skip (filter rows)
#     error (throw an error)
#     keep (put in a special additional bucket)
    
#     NOTE: spark 3.0 will accept multple columns as input/output
#     """
#     encoded_col_name = f'{col_name}_Index'
#     string_indexer = StringIndexer(inputCol=col_name, 
#                                    outputCol=encoded_col_name, 
#                                    handleInvalid='keep')
#     return string_indexer


# In[ ]:





# In[239]:


stages_cat_str_index = [make_string_indexer(c) for c in cat_fields]


# In[243]:


indexOutputCols = [indexer.getOutputCol() for indexer in stages_cat_str_index]
indexOutputCols


# # combine output of StringIndexer and numeric columns

# In[249]:


assemblerInputs = indexOutputCols + numericCols

vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol='features')


# In[246]:


assemblerInputs


# In[248]:


trainDF.select('review_scores_rating', 'review_scores_rating_na').toPandas()


# Combine stages into a pipeline

# Note: unlike `StringIndexer` in Spark 3.0, I have a list of individual `StringIndexer` objects, so need to add as a list

# In[251]:


stages = stages_cat_str_index + [vecAssembler, dt]
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(trainDF)


# check the `maxBins` parameter against our data
# * determines the number of bins into which the continuous featres are discretized or split
# * no `maxBins` param in `scikit-learn` because all the data an dmodel reside on a single machine

# In[253]:


dt.getMaxBins()


# In[255]:


print(dt.explainParams())


# In[256]:


dt.setMaxBins(40)
pipelineModel = pipeline.fit(trainDF)


# ## Extract if-then-else rules learned by the decision tree

# In[257]:


pipelineModel.stages


# In[258]:


dtModel = pipelineModel.stages[-1]


# In[259]:


print(dtModel.toDebugString)


# ## Interpreting Feature Importance
# -----
# hard to know what feature 12 vs 5 is...

# ### Extract feature importance scores

# In[264]:


import pandas as pd

featureImp = (pd.DataFrame(
    list(zip(vecAssembler.getInputCols(), dtModel.featureImportances)),
    columns=['feature', 'importance'])
              .sort_values(by='importance', ascending=False)
             )
featureImp


# ## Apply model to test set

# In[265]:


predDF = pipelineModel.transform(testDF)
predDF.select('features', 'price', 'prediction').orderBy('price', ascending=False).show()


# ## Pitfall
# 
# What if we get a massive Airbnb rental? It was 20 bedrooms and 20 bathrooms. What will a decision tree predict?
# 
# It turns out decision trees cannot predict any values larger than they were trained on. The max value in our training set was $10,000, so we can't predict any values larger than that (or technically any values larger than the )

# In[266]:


from pyspark.ml.evaluation import RegressionEvaluator

regressionEvaluator = RegressionEvaluator(predictionCol="prediction", 
                                          labelCol="price", 
                                          metricName="rmse")

rmse = regressionEvaluator.evaluate(predDF)
r2 = regressionEvaluator.setMetricName("r2").evaluate(predDF)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")


# # Random Forest

# * Ensemble
#     * build many models, and combine/average their predictions
# * Random forest - ensemble of decision tree
#     * Bootstrapping samples by rows
#         * sample with replacement
#     * each tree is trained on a different bootstrap sample of the data set
#     * aggregate the predictions
#     * *bootstrap aggregatine*, or *bagging*
#     * each tree samples the same number of data points with replacement from the original data set
#     * `subsamplingRate` - how many data points to sample for each tree
# * Random feature selection by columns
#     * With bagging, all the trees are highly correlated
#     * For each split, only consider a **random subset of columns**
#         * 1/3 the features for `RandomForeestRegressor`
#         * $\sqrt{\text{num features}}$ for `RandomForestClassifier`
#     * Keep each tree shallow (due to this extra randomness. why?)
#     * each tree is worse than a single decision tree
#     * each tree is a **weak learner**
#     * combining weak learners into an ensemble makes th forest more robust than a single decision tree
# ----
# * Demonstrates power of distributed machine learning
#     * can build each tree independently of others

# In[267]:


from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(labelCol='price', maxBins=40, seed=42)


# ## k-fold Cross Validation

# * What data set to use to optimze hyperparameters?
# * training set -> overfit
# * testing set -> cannot verify how well model generalizes
# * validation set!

# * break the data in to k groups
# * train the model on folds 1-(k-1), and evaluate on fold k
# * train the model on folds 1-(k-2) + k, and evaluate on fold (k-1)
# * repeat k times, each time evaluating on a different fold
# * average the evaluation metrics of the k trials to get an estimate of how it will perform on unseen data

# ## Hyperparameter optimization

# To perform a hyperparameter search in Spark, take the following steps :
# 
# 1. Define the estimator you want to evaluate.
# 
# 2. Specify which hyperparameters you want to vary, as well as their respective values, using the `ParamGridBuilder`.
# 
# 3. Define an `evaluator` to specify which metric to use to compare the various models.
# 
# 4. Use the `CrossValidator` to perform cross-validation, evaluating each of the various models.
# 
# 

# In[268]:


pipeline = Pipeline(stages = stages_cat_str_index + [vecAssembler, rf])


# ### Setup `ParamGrid`

# For our `ParamGridBuilder`, we’ll vary our `maxDepth` to be 2, 4, or 6 and `numTrees` (the number of trees in our random forest) to be 10 or 100. This will give us a grid of 6 (3 x 2) different hyperparameter configurations in total:
# 
# 

# In[270]:


from pyspark.ml.tuning import ParamGridBuilder
paramGrid = (ParamGridBuilder()
            .addGrid(rf.maxDepth, [2, 4, 6])
            .addGrid(rf.numTrees, [10, 100])
            .build())


# ### Setup Evaluator

# Now that we have set up our hyperparameter grid, we need to define how to evaluate each of the models to determine which one performed best. For this task we will use the `RegressionEvaluator`, and we’ll use RMSE as our metric of interest:
# 
# 

# In[271]:


evaluator = RegressionEvaluator(labelCol='price',
                               predictionCol='prediction',
                               metricName='rmse')


# ### Do cross-validation

# * We will perform our k-fold cross-validation using the `CrossValidator`, which accepts an `estimator`, `evaluator`, 
# and `estimatorParamMaps` so that it knows which model to use, how to evaluate the model, 
# and which hyperparameters to set for the model. 
# * We can also set the number of folds we 
# want to split our data into (`numFolds=3`), as well as setting a seed so we have reproducible splits 
# across the folds (`seed=42`). 
# * Let’s then fit this cross-validator to our training data set:
# 
# 

# In[272]:


from pyspark.ml.tuning import CrossValidator

cv = CrossValidator(estimator=pipeline,
                    evaluator=evaluator,
                    estimatorParamMaps=paramGrid,
                    numFolds=3,
                    seed=42)
cvModel = cv.fit(trainDF)


# How many models does this train?
# 
# $19 = (3 \text{ folds}) \times (3 \times 2 \text{ hyperparam configs}) + (1 \text{ final model with optimal params})$

# In[274]:


list(zip(cvModel.getEstimatorParamMaps(), cvModel.avgMetrics))


# In[279]:


p = cvModel.getEstimatorParamMaps()[0]


# In[303]:


pms = [dict(zip([p.name for p in p.keys()], p.values())) for p in cvModel.getEstimatorParamMaps()]


# In[307]:


list(zip(np.round(cvModel.avgMetrics, 2), pms))


# We can see that the best model from our `CrossValidator` (the one with the lowest RMSE) had 
# `maxDepth=6` and `numTrees=100`. However, this took a long time to run. 
# 

# ## Optimizing Pipelines

# ### Parallelism

# * In the preceding code, even though each of the models in the cross-validator is technically independent, `spark.ml` actually trains the collection of models sequentially rather than in parallel. 
# * In Spark 2.3, a `parallelism` parameter was introduced to solve this problem. 
# * This parameter determines the number of models to train in parallel, which themselves are fit in parallel. From the Spark Tuning Guide [https://spark.apache.org/docs/latest/ml-tuning.html]

# The value of `parallelism` should be chosen carefully to maximize parallelism without exceeding cluster resources, and larger values may not always lead to improved performance. 
# 
# Generally speaking, a value up to 10 should be sufficient for most clusters.

# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'cvModel = cv.setParallelism(4).fit(trainDF)')


# In[310]:


get_ipython().run_cell_magic('timeit', '', 'cvModel = cv.setParallelism(1).fit(trainDF)')


# ### Put cross-validator inside the pipeline (instead of pipeline in cross-validator)
# 
# * don't repeat stages that don't change

# In[311]:


from pyspark.ml.tuning import CrossValidator

cv = CrossValidator(estimator=rf,
                    evaluator=evaluator,
                    estimatorParamMaps=paramGrid,
                    numFolds=3,
                    parallelism=4,
                    seed=42)

pipeline = Pipeline(stages=stages_cat_str_index + [vecAssembler, cv])


# In[313]:


get_ipython().run_cell_magic('timeit', '', 'pipelineModel = pipeline.fit(trainDF)')


# In[314]:


predDF = pipelineModel.transform(testDF)

regressionEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regressionEvaluator.evaluate(predDF)
r2 = regressionEvaluator.setMetricName("r2").evaluate(predDF)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")


# # Distributed K-Means

# Use the iris dataset

# In[316]:


from sklearn.datasets import load_iris
import pandas as pd


# Load data, and convert to Spark data frame

# In[317]:


iris = load_iris()


# In[319]:


type(iris)


# In[320]:


iris_pd = pd.concat([pd.DataFrame(iris.data, columns=iris.feature_names),
                     pd.DataFrame(iris.target, columns=['label'])],
                   axis=1)


# In[321]:


irisDF = spark.createDataFrame(iris_pd)


# In[322]:


irisDF.show()


# Notice that we have four values as "features".  We'll reduce those down to two values (for visualization purposes) and convert them to a `DenseVector`.  To do that we'll use the `VectorAssembler`. 

# In[324]:


from pyspark.ml.feature import VectorAssembler

vecAssembler = VectorAssembler(inputCols=['sepal length (cm)', 'sepal width (cm)'],
                               outputCol='features')
irisTwoFeaturesDF = vecAssembler.transform(irisDF)
irisTwoFeaturesDF.show()


# In[326]:


from pyspark.ml.clustering import KMeans

kmeans = KMeans(k=3, seed=221, maxIter=20)

# fit the estimator, and pass in irisTwoFeaturesDF
model = kmeans.fit(irisTwoFeaturesDF)


# In[327]:


# obtain clusterCenters from KMeansModel
centers = model.clusterCenters()

# use the model to transform the DF by adding cluster predictions
transformerDF = model.transform(irisTwoFeaturesDF)

print(centers)


# In[328]:


modelCenters = []
iterations = [0, 2, 4, 7, 10, 20]
for i in iterations:
    kmeans = KMeans(k=3, seed=221, maxIter=i)
    model = kmeans.fit(irisTwoFeaturesDF)
    modelCenters.append(model.clusterCenters())


# In[330]:


print("modelCenters:")
for centroids in modelCenters:
  print(centroids)


# Let's visualize how our clustering performed against the true labels of our data.
# 
# Remember: K-means doesn't use the true labels when training, but we can use them to evaluate. 
# 
# Here, the star marks the cluster center.

# In[332]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def prepareSubplot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999', 
                gridWidth=1.0, subplots=(1, 1)):
    """Template for generating the plot layout."""
    plt.close()
    fig, axList = plt.subplots(subplots[0], subplots[1], figsize=figsize, facecolor='white', 
                               edgecolor='white')
    if not isinstance(axList, np.ndarray):
        axList = np.array([axList])
    
    for ax in axList.flatten():
        ax.axes.tick_params(labelcolor='#999999', labelsize='10')
        for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
            axis.set_ticks_position('none')
            axis.set_ticks(ticks)
            axis.label.set_color('#999999')
            if hideLabels: axis.set_ticklabels([])
        ax.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
        map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
        
    if axList.size == 1:
        axList = axList[0]  # Just return a single axes object for a regular plot
    return fig, axList


# In[335]:


data = irisTwoFeaturesDF.select("features", "label").collect()
features, labels = zip(*data)

x, y = zip(*features)
centers = modelCenters[5]
centroidX, centroidY = zip(*centers)
colorMap = 'Set1'  # was 'Set2', 'Set1', 'Dark2', 'winter'

fig, ax = prepareSubplot(np.arange(-1, 1.1, .4), np.arange(-1, 1.1, .4), figsize=(8,6))
plt.scatter(x, y, s=14**2, c=labels, edgecolors='#8cbfd0', alpha=0.80, cmap=colorMap)
plt.scatter(centroidX, centroidY, s=22**2, marker='*', c='yellow')
cmap = cm.get_cmap(colorMap)

colorIndex = [.5, .99, .0]
for i, (x,y) in enumerate(centers):
    print(cmap(colorIndex[i]))
    for size in [.10, .20, .30, .40, .50]:
        circle1=plt.Circle((x,y),size,color=cmap(colorIndex[i]), alpha=.10, linewidth=2)
        ax.add_artist(circle1)

ax.set_xlabel('Sepal Length'), ax.set_ylabel('Sepal Width');
display(fig);


# In addition to seeing the overlay of the clusters at each iteration, we can see how the cluster centers moved with each iteration (and what our results would have looked like if we used fewer iterations).

# In[336]:


x, y = zip(*features)

oldCentroidX, oldCentroidY = None, None

fig, axList = prepareSubplot(np.arange(-1, 1.1, .4), np.arange(-1, 1.1, .4), figsize=(11, 15),
                             subplots=(3, 2))
axList = axList.flatten()

for i,ax in enumerate(axList[:]):
    ax.set_title('K-means for {0} iterations'.format(iterations[i]), color='#999999')
    centroids = modelCenters[i]
    centroidX, centroidY = zip(*centroids)
    
    ax.scatter(x, y, s=10**2, c=labels, edgecolors='#8cbfd0', alpha=0.80, cmap=colorMap, zorder=0)
    ax.scatter(centroidX, centroidY, s=16**2, marker='*', c='yellow', zorder=2)
    if oldCentroidX and oldCentroidY:
      ax.scatter(oldCentroidX, oldCentroidY, s=16**2, marker='*', c='grey', zorder=1)
    cmap = cm.get_cmap(colorMap)
    
    colorIndex = [.5, .99, 0.]
    for i, (x1,y1) in enumerate(centroids):
      print(cmap(colorIndex[i]))
      circle1=plt.Circle((x1,y1),.35,color=cmap(colorIndex[i]), alpha=.40)
      ax.add_artist(circle1)
    
    ax.set_xlabel('Sepal Length'), ax.set_ylabel('Sepal Width')
    oldCentroidX, oldCentroidY = centroidX, centroidY

plt.tight_layout()

display(fig)


# In[ ]:




