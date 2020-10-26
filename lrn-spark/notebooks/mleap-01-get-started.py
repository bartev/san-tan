#!/usr/bin/env python
# coding: utf-8

# # Description
# ----
# Getting started with mleap
# 
# https://mleap-docs.combust.ml/getting-started/

# ## Typical MLeap Workflow
# 
# A typical MLeap workflow consists of 3 parts:
# 
# 1. Training: Write your ML Pipelines the same way you do today
# 2. Serialization: Serialize all of the data processing (ml pipeline) and the algorithms to Bundle.ML
# 3. Execution: Use MLeap runtime to execute your serialized pipeline without dependencies on Spark or Scikit (you'll still need TensorFlow binaries)
# 

# ## Serialization
# 
# Once you have your pipeline trained, MLeap provides functionality to serialize the entire ML/Data Pipeline and your trained algorithm (linear models, tree-based models, neural networks) to Bundle.ML. Serialization generates something called a `bundle` which is a physical representation of your pipeline and algorithm that you can deploy, share, view all of the pieces of the pipeline.
# 
# 

# ## Execution
# 
# The goal of MLeap was initially to enable scoring of Spark's ML pipelines without the dependency on Spark. That functionality is powered by MLeap Runtime, which loads your serialized bundle and executes it on incoming dataframes (LeapFrames).
# 
# Did we mention that MLeap Runtime is extremely fast? We have recorded benchmarks of micro-second execution on LeapFrames and sub-5ms response times when part of a RESTful API service.
# 
# **Note: As of right now, MLeap runtime is only provided as a Java/Scala library, but we do plan to add python bindings in the future.**
# 
# 

# # Load libraries and data

# In[6]:


# imports for adhoc notebooks
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))

# magics
get_ipython().run_line_magic('load_ext', 'blackcellmagic')
# start cell with `%% black` to format using `black`

get_ipython().run_line_magic('load_ext', 'autoreload')
# start cell with `%autoreload` to reload module
# https://ipython.org/ipython-doc/stable/config/extensions/autoreload.html


# In[2]:


from mleap import pyspark

from pyspark.ml.linalg import Vectors
from mleap.pyspark.spark_support import SimpleSparkSerializer
from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression

from pyspark.sql import SparkSession

import src.spark.utils as uts


# ## Setup spark

# In[40]:


get_ipython().run_line_magic('autoreload', '')
mleap_jars = uts.get_mleap_jars()
# avro_jar = '/Users/bartev/dev/github-bv/san-tan/lrn-spark/references/spark-avro_2.11-4.0.0.jar'
spark = (
    SparkSession
    .builder
    .appName('mleap-ex')
    .config('spark.jars', mleap_jars)
#     .config('spark.jars', f"{mleap_jars},{avro_jar}")
    .getOrCreate()
    )


# In[41]:


spark


# spark.stop()

# # Simple MLeap example
# ----
# https://mleap-docs.combust.ml/py-spark/

# ## Create simple spark pipeline

# ### Imports MLeap serialization functionality for PySpark

# In[69]:


import mleap.pyspark
from mleap.pyspark.spark_support import SimpleSparkSerializer


# ### Import standard PySpark Transformers and packages

# In[70]:


from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import Row


# ### Create a test data frame

# In[74]:


sc = spark.sparkContext


# In[84]:


l = [('Alice', 1), ('Bob', 2)]
rdd = sc.parallelize(l)
Person = Row('name', 'age')
person = rdd.map(lambda r: Person(*r))
df2 = spark.createDataFrame(person)
df2.collect()


# In[83]:


df2.show()


# ### Build a very simple pipeline using two transformers

# In[85]:


string_indexer = StringIndexer(inputCol='name', outputCol='name_string_index')

feature_assembler = VectorAssembler(inputCols=[string_indexer.getOutputCol()],
                                    outputCol='features')

feature_pipeline = [string_indexer, feature_assembler]

featurePipeline = Pipeline(stages=feature_pipeline)

fittedPipeline = featurePipeline.fit(df2)


# In[104]:


type(fittedPipeline)


# In[103]:


get_ipython().run_line_magic('pinfo2', 'fittedPipeline.serializeToBundle')


# ### Serialize to zip file
# ----
# In order to serialize to a zip file, make sure the URI begins with jar:file and ends with a .zip.
# 
# For example `jar:file:/tmp/mleap-bundle.zip.`
# 
# 

# In[92]:


import os
current_path = os.getcwd()


# In[93]:


current_path


# In[100]:


get_ipython().run_line_magic('autoreload', '')
uts.create_mleap_fname('foo.zip', '.')


# In[101]:


fname = uts.create_mleap_fname('pyspark.example.zip', '.')
print(fname)
fittedPipeline.serializeToBundle(fname,
                                fittedPipeline.transform(df2))


# # Databricks AirBnB example

# df = spark.read.format("com.databricks.spark.avro").load("file:////tmp/airbnb.avro")

# In[105]:


df = (
    spark.read.format("csv")
    .option("header", "true")
    .option('inferSchema','true')
    .load("/Users/bartev/dev/gitpie/mleap-demo/data/airbnb.csv")
)


# In[106]:


df.columns


# In[107]:


df.show()


# In[108]:


df.dtypes


# In[109]:


datasetFiltered = (df.filter("price >= 50")
    .filter("price <= 750")
    .filter('bathrooms > 0.0'))


# In[110]:


print(df.count())
print(datasetFiltered.count())


# In[111]:


datasetFiltered.columns


# In[112]:


datasetFiltered.registerTempTable('df')

q = """
select
    id,
    case when state in ('NY', 'CA', 'London', 'Berlin',
        'TX', 'IL', 'OR', 'DC', 'WA')
        then state
        else 'Other'
        end as state,
    price,
    bathrooms,
    bedrooms,
    room_type,
    host_is_superhost,
    cancellation_policy,
    case when security_deposit is null
        then 0.0
        else security_deposit
        end as security_deposit,
    price_per_bedroom,
    case when number_of_reviews is null
        then 0.0
        else number_of_reviews
        end as number_of_reviews,
    case when extra_people is null
        then 0.0
        else extra_people
        end as extra_people,
    instant_bookable,
    case when cleaning_fee is null
        then 0.0
        else cleaning_fee
        end as cleaning_fee,
    case when review_scores_rating is null
        then 0.0
        else review_scores_rating
        end as review_scores_rating,
    case when square_feet is not null and square_feet > 100
        then square_feet
        when (square_feet is null or square_feet <=100)
            and (bedrooms is null or bedrooms = 0)
        then 350.0
        else 380 * bedrooms        
        end as square_feet,
    case when bathrooms >= 2
        then 1.0
        else 0.0
        end as n_bathrooms_more_than_two
from df
where bedrooms is not null
        """
datasetImputed = spark.sql(q)


# In[113]:


(
    datasetImputed.select(
        "square_feet", "price", "bedrooms", "bathrooms", "cleaning_fee"
    )
    .describe()
    .show()
)


# ## Look at some summary statistics of the data

# In[114]:


# most popular states

spark.sql("""
select
    state,
    count(*) as n,
    cast(avg(price) as decimal(12,2)) as avg_price,
    max(price) as max_price
from df
group by state
order by n desc
""").show()


# In[116]:


datasetImputed.limit(10).toPandas()


# ## Define continous and categorical features

# In[144]:


continuous_features = [
    "bathrooms",
    "bedrooms",
    "security_deposit",
    "cleaning_fee",
    "extra_people",
    "number_of_reviews",
    "square_feet",
    "review_scores_rating",
]
categorical_features = [
    "state",
    "room_type",
    "host_is_superhost",
    "cancellation_policy",
    "instant_bookable",
]

all_features = continuous_features + categorical_features


# In[145]:


dataset_imputed = datasetImputed.persist()


# ## Split data into train and validation

# In[146]:


[training_dataset, validation_dataset] = dataset_imputed.randomSplit([0.7, 0.3])


# ## Continuous feature pipeline

# In[147]:


continuous_feature_assembler = VectorAssembler(
    inputCols=continuous_features, outputCol="unscaled_continuous_features"
)

continuous_feature_scaler = StandardScaler(
    inputCol="unscaled_continuous_features",
    outputCol="scaled_continuous_features",
    withStd=True,
    withMean=True,
)


# ## Categorical features pipeline

# In[148]:


import src.models.train_model as tm


# In[149]:


get_ipython().run_line_magic('autoreload', '')
categorical_feature_indexers = tm.make_string_indexer_list(categorical_features)

# ohe_input_cols = tm.get_output_col_names(categorical_feature_indexers)

categorical_feature_ohe = tm.make_one_hot_encoder(categorical_features)


# In[150]:


categorical_feature_ohe.getInputCols()


# In[151]:


categorical_feature_ohe.getOutputCols()


# ## Assemble features and feature pipeline

# In[152]:


estimatorLr = (
    [continuous_feature_assembler, continuous_feature_scaler]
    + categorical_feature_indexers
    + [categorical_feature_ohe]
)

featurePipeline = Pipeline(stages=estimatorLr)

sparkFeaturePipelineModel = featurePipeline.fit(dataset_imputed)

print("Finished constructing the pipeline")


# ## Train a Linear Regression Model

# In[158]:


linearRegression = LinearRegression(featuresCol='scaled_continuous_features',
                                   labelCol='price',
                                   predictionCol='price_prediction',
                                   maxIter=10,
                                   regParam=0.3,
                                   elasticNetParam=0.8)

pipeline_lr = [sparkFeaturePipelineModel, linearRegression]

sparkPipelineEstimatorLr = Pipeline(stages=pipeline_lr)

sparkPipelineLr = sparkPipelineEstimatorLr.fit(dataset_imputed)

print('Complet: Training Linear Regression')


# ## Train a Logistic Regression Model

# In[163]:


logisticRegression = LogisticRegression(featuresCol='scaled_continuous_features',
                                       labelCol='n_bathrooms_more_than_two',
                                       predictionCol='n_bathrooms_more_than_two_prediction',
                                       maxIter=10)

pipeline_log_r = [sparkFeaturePipelineModel, logisticRegression]

sparkPipelineEstimatorLogr = Pipeline(stages=pipeline_log_r)

sparkPipelineLogr = sparkPipelineEstimatorLogr.fit(dataset_imputed)

print('Complete: Training Logistic Regression')


# ## Serialize the model ot Bundle.ML

# In[165]:


fname_lr = uts.create_mleap_fname('pyspark.lr.zip', '../models/')
fname_logr = uts.create_mleap_fname('pyspark.logr.zip', '../models/')


# In[168]:


for mod, fname in [(sparkPipelineLr, fname_lr),
                   (sparkPipelineLogr, fname_logr)]:
    mod.serializeToBundle(fname, mod.transform(dataset_imputed))


# In[170]:


ls ../models/


# ## (Optional) Deserialize from Bundle.ML

# In[171]:


sparkPipelineLR_des = PipelineModel.deserializeFromBundle(fname_lr)


# In[172]:


type(sparkPipelineLR_des)


# In[ ]:




