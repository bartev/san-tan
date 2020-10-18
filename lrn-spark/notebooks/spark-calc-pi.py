#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark
import random


# In[22]:


sc


# In[23]:


spark


# In[29]:


num_samples = 100000000
def inside(p):
    x, y = random.random(), random.random()
    return (x * x) + (y * y) < 1


# In[30]:


get_ipython().run_cell_magic('timeit', '', 'count = sc.parallelize(range(0, num_samples)).filter(inside).count()')


# In[28]:


pi = 4 *  count / num_samples
print(pi)
print(f'{count:0,}')


# In[21]:


spark


# # Create a data frame

# ## `seq` and `toDF`

# In[33]:


import pyspark.sql.functions as F
import pyspark.sql


# In[ ]:




