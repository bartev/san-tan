
import pymongo
from pymongo import MongoClient

# Get mongo client with default values
client = MongoClient()

# Get a db called tutorial
db = client.tutorial
db.collection_names()

import datetime

# Get a collection in db
collection = db.numbers
collection = db.test_collection

# Create a post item to add
post = {'author': 'Mike',
	'text':'My first blog post',
	'tags':['mongodb', 'python', 'pymongo'],
	'date':datetime.datetime.utcnow()}

# Add the collection posts to db
posts = db.posts

# Insert the post from above in posts
post_id = posts.insert(post)

# Do some simple queries on the posts collection
posts.find_one({'_id':post_id})
posts.find({'author':'Mike'})
posts.find_one({'author':'Mike'})
posts.find()
posts.find_one()


# Add more posts
new_posts = [{"author": "Mike",
               "text": "Another post!",
               "tags": ["bulk", "insert"],
               "date": datetime.datetime(2009, 11, 12, 11, 14)},
              {"author": "Eliot",
               "title": "MongoDB is fun",
               "text": "and pretty easy too!",
               "date": datetime.datetime(2009, 11, 10, 10, 45)}]

# Bulk insert
posts.insert(new_posts)

# result of find() is a cursor object
tmp = posts.find()
# >>> tmp.find({'author':'Mike'})
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# AttributeError: 'Cursor' object has no attribute 'find'


d = datetime.datetime(2009, 11, 12, 12)
for post in posts.find({'date':{'$lt':d}}).sort('author'):
	print post

