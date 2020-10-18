#!/usr/bin/env python
# coding: utf-8

# # Description
# 
# learn pyspark basics from a blog
# 
# Follow this blogpost https://towardsdatascience.com/the-hitchhikers-guide-to-handle-big-data-using-spark-90b9be0fe89a
# 
# 
# * RDD Programming Guide http://spark.apache.org/docs/latest/rdd-programming-guide.html#actions
# 
# Follow up
# 
# See coursera courses
# * Big Data Essentials: HDFS, MapReduce and Spark RDD
#     * https://www.coursera.org/learn/big-data-analysis?ranMID=40328&ranEAID=lVarvwc5BD0&ranSiteID=lVarvwc5BD0-Y2ZYRU0eP2qngfq6ffGH2Q&siteID=lVarvwc5BD0-Y2ZYRU0eP2qngfq6ffGH2Q&utm_content=2&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0
# * Big Data Analysis: Hive, Spark SQL, DataFrames and GraphFrames
#     * https://www.coursera.org/learn/big-data-analysis?ranMID=40328&ranEAID=lVarvwc5BD0&ranSiteID=lVarvwc5BD0-Y2ZYRU0eP2qngfq6ffGH2Q&siteID=lVarvwc5BD0-Y2ZYRU0eP2qngfq6ffGH2Q&utm_content=2&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0

# # Imports

# In[1]:


import pyspark


# # Setup spark context

# In[2]:


sc


# In[3]:


sc = pyspark.SparkContext('local[*]')


# In[4]:


sc


# In[35]:


sc.environment


# In[38]:


sc.applicationId


# In[39]:


sc.appName


# In[40]:


sc.version


# In[41]:


sc.pythonVer


# In[43]:


sc.sparkHome


# In[46]:


sc.startTime


# In[48]:


sc.startTime


# # Read in Macbeth

# In[5]:


ls


# In[8]:


lines = sc.textFile('ShakespearePlaysPlus/tragedies/Macbeth.txt')


# In[9]:


print(lines.count())


# In[18]:


# Create a list with all words
# create tuple (word, 1)
# reduce by key (i.e. the word)

counts = (lines.flatMap(lambda x: x.replace('\x00', '').split(' '))
           .map(lambda x: (x, 1))
           .reduceByKey(lambda x, y : x + y))


# In[19]:


counts


# In[22]:


# get the output on local
output = counts.take(10)

output


# In[23]:


# print output
for word, count in output:
  print(f'{word}: {count:d}')


# # Map

# In[24]:


my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Square every term in my_list
squared_list = map(lambda x: x**2, my_list)
print(list(squared_list))


# # filter

# In[25]:


# keep only even numbers

filtered_list = filter(lambda x: x%2 == 0, my_list)
print(list(filtered_list))


# # reduce

# In[26]:


import functools
my_list = [1,2,3,4,5]

# sum all elements in my_list
sum_list = functools.reduce(lambda x,y: x + y, my_list)
sum_list


# # RDD

# ## parallelize

# In[28]:


data = [1,2,3,4,5,6,7,8,9,10]

new_rdd = sc.parallelize(data, 4)
new_rdd


# # Two operations - Transformation and Action
# 
# * Transformation - create new dataset from existing RDD
# *  Action - mechanism to get results out of sparkj

# In[32]:


type(lines)


# In[33]:


type(counts)


# In[34]:


type(output)


# # Transformations
# 
# http://spark.apache.org/docs/latest/rdd-programming-guide.html#transformations

# ## Map

# In[58]:


data = [1,2,3,4,5,6,7,8,9,10]
rdd = sc.parallelize(data, 4)
squared_rdd = rdd.map(lambda x: x ** 2)

squared_rdd.collect()


# In[59]:


squared_rdd


# ## Filter
# 
# Return those elements that fulfill the condition

# In[61]:


data = [1,2,3,4,5,6,7,8,9,10]
rdd = sc.parallelize(data, 4)
filtered_rdd = rdd.filter(lambda x: x % 2 == 0)
filtered_rdd.collect()


# ## Distinct
# 
# return only distinct elemens of an RDD

# In[66]:


data = [1,2,2,2,2,3,3,3,3,4,5,6,7,7,7,8,8,8,9,10]
rdd = sc.parallelize(data, 4)
distinct_rdd = rdd.distinct()
distinct_rdd.collect()


# ## flatmap
# 
# Similar to map, but each input item can be mapped to 0 or more output items.
# 
# 

# In[67]:


data = [1,2,3,4]
rdd = sc.parallelize(data, 4)
flat_rdd = rdd.flatMap(lambda x: [x, x**3])
flat_rdd.collect()


# ## Reduce By Key
# 
# The parallel to the reduce in Hadoop MapReduce.
# 
# Now Spark cannot provide the value if it just worked with Lists.
# 
# In Spark, there is a concept of pair RDDs that makes it a lot more flexible. Let's assume we have a data in which we have a product, its category, and its selling price. We can still parallelize the data.

# In[68]:


data = [('Apple','Fruit',200),
        ('Banana','Fruit',24),
        ('Tomato','Fruit',56),
        ('Potato','Vegetable',103),
        ('Carrot','Vegetable',34)]

rdd = sc.parallelize(data, 4)


# Right now our RDD rdd holds tuples.
# 
# Now we want to find out the total sum of revenue that we got from each category.
# 
# To do that we have to transform our rdd to a pair rdd so that it only contains key-value pairs/tuples.

# In[69]:


category_price_rdd = rdd.map(lambda x: (x[1], x[2]))
category_price_rdd.collect()


# Here we used the map function to get it in the format we wanted. When working with textfile, the RDD that gets formed has got a lot of strings. We use map to convert it into a format that we want.
# 
# So now our category_price_rdd contains the product category and the price at which the product sold.
# 
# Now we want to reduce on the key category and sum the prices. We can do this by:

# In[70]:


category_total_price_rdd = category_price_rdd.reduceByKey(lambda x, y: x + y)
category_total_price_rdd.collect()


# ## Group By Key
# 
# Similar to reduceByKey but does not reduces just puts all the elements in an iterator. For example, if we wanted to keep as key the category and as the value all the products we would use this function.
# 
# Let us again use map to get data in the required form.

# In[73]:


ata = [('Apple','Fruit',200),
       ('Banana','Fruit',24),
       ('Tomato','Fruit',56),
       ('Potato','Vegetable',103),
       ('Carrot','Vegetable',34)]
rdd = sc.parallelize(data, 4)
category_product_rdd = rdd.map(lambda x: (x[1], x[0]))
category_product_rdd.collect()


# We then use groupByKey as:

# In[74]:


grouped_products_by_category_rdd = category_product_rdd.groupByKey()
findata = grouped_products_by_category_rdd.collect()
for data in findata:
    print(data[0], list(data[1]))


# In[75]:


findata


# Here the groupByKey function worked and it returned the category and the list of products in that category.

# # Action Basics
# 
# http://spark.apache.org/docs/latest/rdd-programming-guide.html#actions

# You have filtered your data, mapped some functions on it. Done your computation.
# 
# Now you want to get the data on your local machine or save it to a file or show the results in the form of some graphs in excel or any visualization tool.
# 
# You will need actions for that. A comprehensive list of actions is provided here http://spark.apache.org/docs/latest/rdd-programming-guide.html#actions.
# 
# Some of the most common actions that I tend to use are:
# 

# ## collect
# We have already used this action many times. It takes the whole RDD and brings it back to the driver program.

# ## reduce
# 
# Aggregate the elements of the dataset using a function func (which takes two arguments and returns one). The function should be commutative and associative so that it can be computed correctly in parallel.

# In[79]:


rdd = sc.parallelize([1,2,3,4,5])
rdd.reduce(lambda x,y : x+y)


# ## take
# Sometimes you will need to see what your RDD contains without getting all the elements in memory itself. take returns a list with the first n elements of the RDD.

# In[80]:


rdd = sc.parallelize([1,2,3,4,5])
rdd.take(3)


# ## takeOrdered
# takeOrdered returns the first n elements of the RDD using either their natural order or a custom comparator.

# In[82]:


rdd = sc.parallelize([5,3,12,23])
# descending order
rdd.takeOrdered(3, lambda s: -1*s)


# In[83]:


rdd = sc.parallelize([(5,23),(3,34),(12,344),(23,29)])
# descending order
rdd.takeOrdered(3, lambda s: -1*s[1])


# # Understanding The WordCount Example

# Now we sort of understand the transformations and the actions provided to us by Spark.
# 
# It should not be difficult to understand the wordcount program now. Let us go through the program line by line.
# 
# The first line creates an RDD and distributes it to the workers.

# In[84]:


lines = sc.textFile('ShakespearePlaysPlus/tragedies/Macbeth.txt')


# This RDD lines contains a list of sentences in the file. You can see the rdd content using take

# In[85]:


lines.take(5)


# This next line is actually the workhorse function in the whole script.

# In[86]:


counts = (lines.flatMap(lambda x: x.replace('\x00', '').split(' '))
                  .map(lambda x: (x, 1))
                  .reduceByKey(lambda x, y : x + y))


# It contains a series of transformations that we do to the lines RDD. First of all, we do a flatmap transformation.
# 
# The flatmap transformation takes as input the lines and gives words as output. So after the flatmap transformation, the RDD is of the form:
# 
# `['word1','word2','word3','word4','word3','word2']`
# 

# Next, we do a map transformation on the flatmap output which converts the RDD to :
# 
# `[('word1',1),('word2',1),('word3',1),('word4',1),('word3',1),('word2',1)]`
# 

# Finally, we do a reduceByKey transformation which counts the number of time each word appeared.
# 
# After which the RDD approaches the final desirable form.
# 
# `[('word1',1),('word2',2),('word3',2),('word4',1)]`
# 

# This next line is an action that takes the first 10 elements of the resulting RDD locally.

# In[87]:


output = counts.take(10)


# This line just prints the output

# In[88]:


# print output
for word, count in output:
  print(f'{word}: {count:d}')


# And that is it for the wordcount program. Hope you understand it now.
# 
# So till now, we talked about the Wordcount example and the basic transformations and actions that you could use in Spark. But we don’t do wordcount in real life.
# 
# We have to work on bigger problems which are much more complex. Worry not! Whatever we have learned till now will let us do that and more

# # Spark in Action with Example

# Let us work with a concrete example which takes care of some usual transformations.
# 
# We will work on Movielens ml-100k.zip dataset which is a stable benchmark dataset. 100,000 ratings from 1000 users on 1700 movies. Released 4/1998.
# 
# The Movielens dataset contains a lot of files but we are going to be working with 3 files only:
# 
# 1) Users: This file name is kept as “u.user”, The columns in this file are:
# 
#     ['user_id', 'age', 'sex', 'occupation', 'zip_code']
# 
# 2) Ratings: This file name is kept as “u.data”, The columns in this file are:
# 
#     ['user_id', 'movie_id', 'rating', 'unix_timestamp']
# 
# 3) Movies: This file name is kept as “u.item”, The columns in this file are:
# 
#     ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url', and 18 more columns.....]
# 
# Let us start by importing these 3 files into our spark instance using ‘Import and Explore Data’ on the home tab.
# 

# import the files u.user, u.data, u.item

# ## Find 25 most rated movies

# Our business partner now comes to us and asks us to find out the 25 most rated movie titles from this data. How many times a movie has been rated?
# 
# Let us load the data in different RDDs and see what the data contains.

# In[89]:


ls


# In[90]:


userRDD = sc.textFile('Data-ML-100k--master/ml-100k/u.user') 
ratingRDD = sc.textFile('Data-ML-100k--master/ml-100k/u.data') 
movieRDD = sc.textFile('Data-ML-100k--master/ml-100k/u.item') 

print("userRDD:", userRDD.take(1))
print("ratingRDD:", ratingRDD.take(1))
print("movieRDD:", movieRDD.take(1))


# We note that to answer this question we will need to use the ratingRDD. But the ratingRDD does not have the movie name.
# 
# So we would have to merge movieRDD and ratingRDD using movie_id.
# 

# In[92]:


# Create a RDD from RatingRDD that only contains the two columns of interest i.e. movie_id,rating.

RDD_movie_rating = ratingRDD.map(lambda x: (x.split('\t')[1], x.split('\t')[2]))
print(f'RDD_movie_rating: {RDD_movie_rating.take(4)}')


# In[96]:


# Create a RDD from MovieRDD that only contains the two columns of interest i.e. movie_id,title.
RDD_movie_title = movieRDD.map(lambda x: (x.split('|')[0], x.split('|')[1]))
print(f'RDD_movie_title: {RDD_movie_title.take(2)}')


# In[98]:


# merge these two pair RDDs based on movie_id. For this we will use the transformation leftOuterJoin(). See the transformation document.
rdd_movie_title_rating = RDD_movie_rating.leftOuterJoin(RDD_movie_title)
print(f'rdd_movie_title_rating: {rdd_movie_title_rating.take(1)}')


# In[99]:


# use the RDD in previous step to create (movie,1) tuple pair RDD
rdd_title_rating = rdd_movie_title_rating.map(lambda x: (x[1][1], 1))
print(f'rdd_title_rating: {rdd_title_rating.take(1)}')


# In[100]:


# Use the reduceByKey transformation to reduce on the basis of movie_title
rdd_title_rating_cnt = rdd_title_rating.reduceByKey(lambda x, y: x + y)
print(f'rdd_title_rating_cnt: {rdd_title_rating_cnt.take(2)}')


# In[105]:


# Get the final answer by using takeOrdered Transformation
print('#' * 25)
print('25 most rated movies')
[print(x[1], x[0]) for x in rdd_title_rating_cnt.takeOrdered(25, lambda x: -x[1])]
print('#' * 25)


# Star Wars is the most rated movie in the Movielens Dataset.
# 
# Now we could have done all this in a single command using the below command but the code is a little messy now.
# 
# I did this to show that you can use chaining functions with Spark and you could bypass the process of variable creation.
# 

# ### Rewrite as chained function

# In[113]:


print((ratingRDD.map(lambda x : (x.split("\t")[1],x.split("\t")[2])))
       .leftOuterJoin(movieRDD.map(lambda x : (x.split("|")[0],x.split("|")[1])))
       .map(lambda x: (x[1][1],1))
       .reduceByKey(lambda x,y: x+y)
      .takeOrdered(25,lambda x:-x[1]))


# ## Find 25 most highly rated movies

# Let us do one more. For practice:
# 
# Now we want to find the most highly rated 25 movies using the same dataset. We actually want only those movies which have been rated at least 100 times.
# 

# In[118]:


rdd_movie_title_rating.first()


# In[125]:


# We create an RDD that contains sum of all the ratings for a particular movie
rdd_title_ratings_sum = (
    rdd_movie_title_rating
    .map(lambda x: (x[1][1], int(x[1][0])))
    .reduceByKey(lambda x, y: x + y))
print(f'rdd_title_ratings_sum: {rdd_title_ratings_sum.take(2)}')


# In[134]:


# Merge this data with the RDD rdd_title_ratingcnt we created in the last step
# And use Map function to divide ratingsum by rating count.

rdd_title_ratings_mean_rating_count = (
    rdd_title_ratings_sum
    .leftOuterJoin(rdd_title_rating_cnt)
    .map(lambda x: (x[0], x[1][0] / x[1][1], x[1][1]))
)

print(f'rdd_title_ratings_mean_rating_count: {rdd_title_ratings_mean_rating_count.take(1)}')


# In[137]:


# We could use take ordered here only but we want to only get the movies which have count
# of ratings more than or equal to 100 so lets filter the data RDD.
rdd_title_rating_rating_ct_gt_100 = (rdd_title_ratings_mean_rating_count
    .filter(lambda x: x[2] >= 100))

print(f'rdd_title_rating_rating_ct_gt_100: {rdd_title_rating_rating_ct_gt_100.take(1)}')


# In[138]:


# Get the final answer by using takeOrdered Transformation
print('#' * 25)
print('25 highly rated movies')
[print(x[1], x[0]) for x in rdd_title_rating_rating_ct_gt_100.takeOrdered(25, lambda x: -x[1])]
print('#' * 25)


# We have talked about RDDs till now as they are very powerful.
# 
# You can use RDDs to work with non-relational databases too.
# 
# They let you do a lot of things that you couldn’t do with SparkSQL?
# 
# Yes, you can use SQL with Spark too which I am going to talk about now.

# # Spark DataFrames
# 
# https://docs.databricks.com/spark/latest/dataframes-datasets/introduction-to-dataframes-python.html#

# Spark has provided DataFrame API for us Data Scientists to work with relational data. Here is the documentation for the adventurous folks.
# 
# Remember that in the background it still is all RDDs and that is why the starting part of this post focussed on RDDs.
# 
# I will start with some common functionalities you will need to work with Spark DataFrames. Would look a lot like Pandas with some syntax changes.
# 

# ## Setup `SparkSession`

# In[143]:


from pyspark.sql import SparkSession


# In[144]:


spark = (SparkSession
        .builder
        .appName('Python Spark SQL basic example')
        .config('spark.some.config.option', 'some-value')
        .getOrCreate())


# In[145]:


spark


# ## Reading the file

# In[147]:


ratings = spark.read.load('Data-ML-100k--master/ml-100k/u.data', 
                          format='csv',
                          sep='\t',
                          inferSchema='true',
                          header='false')


# ## Show File

# In[148]:


ratings.show()


# In[149]:


display(ratings)


# ## Change column names

# In[150]:


get_ipython().set_next_input('ratings = ratings.toDF');get_ipython().run_line_magic('pinfo2', 'ratings.toDF')


# In[151]:


ratings = ratings.toDF(*['user_id', 'movie_id', 'rating', 'unix_timestamp'])
display(ratings)


# In[152]:


ratings.show()


# ## Some basic stats

# In[154]:


print(f'row count:    {ratings.count()}') 
print(f'column count: {len(ratings.columns)}')


# In[156]:


ratings.describe().show()


# In[157]:


ratings.describe().toPandas()


# ## Select a few columns

# In[158]:


ratings.select('user_id', 'movie_id').show()


# In[160]:


ratings.filter((ratings.rating==5) & (ratings.user_id==253)).show()


# ## Groupby

# We can use groupby function with a spark dataframe too. Pretty much same as a pandas groupby with the exception that you will need to import `pyspark.sql.functions`
# 

# Here we have found the count of ratings and average rating from each user_id

# In[165]:


from pyspark.sql import functions as F

ratings.groupBy('user_id').agg(F.count('user_id'), F.mean('rating')).show()


# ## sort

# In[166]:


ratings.sort('user_id').show()


# ## Descending sort

# In[171]:


from pyspark.sql import functions as F

ratings.sort(F.desc('user_id'), F.desc('rating'), 'movie_id').show()


# ## Joins/Merging with Spark Dataframes

# I was not able to find a pandas equivalent of merge with Spark DataFrames but we can use SQL with dataframes and thus we can merge dataframes using SQL.
# 
# Let us try to run some SQL on Ratings.
# 
# We first register the ratings df to a temporary table ratings_table on which we can run sql operations.
# 
# As you can see the result of the SQL select statement is again a Spark Dataframe.

# ## Define `sqlContext`

# In[174]:


from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)


# In[175]:


ratings.registerTempTable('ratings_table')
newDf = sqlContext.sql('select * from ratings_table where rating > 4')
newDf.show()


# In[176]:


newDf.printSchema()


# Let us now add one more Spark Dataframe to the mix to see if we can use join using the SQL queries:
# 

# In[177]:


# get one more df to join
movies = spark.read.load('Data-ML-100k--master/ml-100k/u.item', 
                        format='csv', sep='|', inferSchema='true', header='false')
movies.show()


# In[178]:


# change column names
movies = movies.toDF(*['movie_id', 'movie_title',
                       'release_date',
                       'video_release_date','IMDb_URL','unknown','Action','Adventure',
                       'Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy',
                       'Film_Noir','Horror','Musical','Mystery','Romance','Sci_Fi','Thriller','War','Western'])


# In[180]:


movies.toPandas()


# In[181]:


movies.registerTempTable('movies_table')

q = """
select ratings_table.*,
    movies_table.movie_title
from ratings_table
left join movies_table on movies_table.movie_id = ratings_table.movie_id"""
sqlContext.sql(q).show()


# Let us try to do what we were doing earlier with the RDDs. Finding the top 25 most rated movies:
# 

# In[183]:


q = """
select movie_id,
    movie_title,
    count(user_id) as num_ratings
from (select r.*, m.movie_title
    from ratings_table r
    left join movies_table m on m.movie_id = r.movie_id) A
group by movie_id, movie_title
order by num_ratings desc
"""

sqlContext.sql(q).show()


# And finding the top 25 highest rated movies having more than 100 votes:
# 

# In[188]:


q = """
select movie_id,
    movie_title,
    avg(rating) as avg_rating,
    count(movie_id) as num_ratings
from (select r.*, m.movie_title
    from ratings_table r
    left join movies_table m on m.movie_id = r.movie_id) A
group by movie_id, movie_title
having num_ratings > 100

order by avg_rating desc

"""

high_rated_ddf = sqlContext.sql(q)
high_rated_ddf.show()


# # Display

# How do I get this to work in my notebook?

# # Converting from Spark Dataframe to RDD and vice versa

# Sometimes you may want to convert to RDD from a spark Dataframe or vice versa so that you can have the best of both worlds.

# ## To convert from DF to RDD, you can simply do :

# In[191]:


high_rated_ddf.rdd.take(2)


# In[193]:


high_rated_ddf.head(2)


# ## To go from an RDD to a dataframe

# In[194]:


from pyspark.sql import Row

# create an RDD first
data = [('A',1),('B',2),('C',3),('D',4)]
rdd = sc.parallelize(data)


# In[195]:


rdd


# In[196]:


# map the schema using Row
rdd_new = rdd.map(lambda x: Row(key=x[0], value=int(x[1])))


# In[200]:


# convert the rdd to a DataFrame
rdd_as_df = sqlContext.createDataFrame(rdd_new)
rdd_as_df.show()


# In[ ]:




