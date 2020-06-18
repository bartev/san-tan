#!/usr/bin/env python
# coding: utf-8

# # Description
# 
# * techniques to clean data
# 
# 
# -----
# 
# * filter examples here https://hackingandslacking.com/cleaning-pyspark-dataframes-1a3f5fdcedd1

# # Imports/setup

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# In[2]:


import os
import pyspark
import pandas as pd


# In[3]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

pd.set_option('display.float_format', '{:.2f}'.format)


# In[4]:


import pyspark
sc = pyspark.SparkContext('local', appName='clean-data')


# In[5]:


print(f'sc.environment:   {sc.environment}')
print(f'sc.applicationId: {sc.applicationId}')
print(f'sc.appName:       {sc.appName}')
print(f'sc.version:       {sc.version}')
print(f'sc.pythonVer:     {sc.pythonVer}')
print(f'sc.sparkHome:     {sc.sparkHome}')
print(f'sc.startTime:     {sc.startTime}')


# In[6]:


from pyspark.sql import SparkSession

spark = (SparkSession
            .builder
            .appName('MAFIA pyspark sql session')
            .config('spark.some.config.option', 'some-value')
            .getOrCreate())


# # Dummy data

# In[21]:


df = (spark.read
      .option('header',True)
      .csv('dirty-data.csv'))
df.show()


# In[22]:


df.dtypes


# In[23]:


from pyspark.sql.functions import col


# In[31]:


df2 = (df.select(
    col('name'), 
    col('age').cast('float'),
    col('person').cast('boolean'))
)

print(df2.dtypes)
df2.show()


# In[35]:


bool_feats = ['person']
num_feats = ['age']
cat_feats = ['name']


# In[92]:


[col(c).cast('string').alias(f'{c}_cat') if c in bool_feats + cat_feats
                   else col(c).cast('float').alias(f'{c}_fl') if c in num_feats
                   else col(c).cast('string').alias(f'{c}_foo')
            for c in bool_feats + num_feats + cat_feats]


# In[94]:


df3 = df2.select([col(c).cast('string').alias(f'{c}_cat') if c in bool_feats + cat_feats
                   else col(c).cast('float').alias(f'{c}_fl') if c in num_feats
                   else col(c).cast('string').alias(f'{c}_foo')
            for c in bool_feats + num_feats + cat_feats])

print(df3.dtypes)
df3.show()


# In[49]:


from pyspark.sql.functions import expr


# In[39]:


# this is harder to program for varying lists
df4 = (df2.select(
    col('person').cast('string'),
    col('name').cast('string'),
    col('age').cast('float')))
print(df4.dtypes)
df4.show()


# I like this

# In[93]:


def gen_select_items(feat_list, feat_type):
    """return a list of casted items to put in select call"""
    return [col(c).cast(feat_type) for c in feat_list]
    
def gen_select_all(cat_feats, num_feats):
    """return a list of columns to select, each cast as a string or float"""
    cat_feat_list = gen_select_items(cat_feats, 'string')
    num_feat_list = gen_select_items(num_feats, 'float')
    return cat_feat_list + num_feat_list


# In[85]:


sel_items = gen_select_all(cat_feats=cat_feats + bool_feats,
               num_feats=num_feats)
sel_items


# In[86]:


df5 = df2.select(sel_items)
print(df4.dtypes)
df5.show()


# In[88]:


df6 = (df2.select(
    col('person').cast(pyspark.sql.types.StringType),
    col('name').cast('string'),
    col('age').cast('float')))
print(df6.dtypes)
df6.show()


# # Filtering

# In[59]:


df5.dropna().show()


# In[62]:


df5.filter(df5.age < 10).show()


# In[65]:


cond = df5.age < 10
df5.filter(cond).show()


# In[67]:


df5.filter(col('age') < 10).show()


# In[73]:


df5.filter(col('name').contains('j') | col('name').contains('d')).show()


# In[77]:


df5.filter(col('name').like('%j%')).show()


# In[80]:


df5.filter(col('age').isin(10.0, 12)).show()


# In[83]:


df5.orderBy('person', 'name', ascending=[1,0]).show()


# # schemas - doesn't work

# In[95]:


df5.printSchema()


# In[103]:


list(map(lambda x: {x.name: x}, df5.schema.fields))


# In[ ]:





# In[116]:


raw_1 = (spark.read
      .option('header',True)
         .schema(df5.schema.simpleString())
      .csv('dirty-data.csv'))


# In[111]:


df5.schema.fieldNames()


# In[115]:


df5.schema.simpleString()


# In[108]:


df


# In[119]:


[i for i in df5.schema]


# In[120]:


df5.schema


# In[ ]:




