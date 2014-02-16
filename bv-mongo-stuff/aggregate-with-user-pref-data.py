# Data model

import pymongo
import datetime
dt = datetime.datetime

input_data = [
	{ '_id' : "jane",
	  'joined' : dt(2011, 3, 2),
	  'likes' : ["golf", "racquetball"]},
	{ '_id' : "joe",
	  'joined' : dt(2012, 7, 5),
	  'likes' : ["tennis", "golf", "swimming"]},
	{ '_id' : "jill",
	  'joined' : dt(2011, 1, 31),
	  'likes' : ["tennis", "bicycling", "horseback riding"]},
	{ '_id' : "jack",
	  'joined' : dt(2011, 1, 31),
	  'likes' : ["golf", "swimming"]}
	]

# Get a db called test
client = MongoClient()
db = client.test
users = db.users
users.insert(input_data)
# ['jane', 'joe']
for user in users.find(): user
# {u'_id': u'jane', u'joined': datetime.datetime(2011, 3, 2, 0, 0), u'likes': [u'golf', u'racquetball']}
# {u'_id': u'joe', u'joined': datetime.datetime(2012, 7, 2, 0, 0), u'likes': [u'tennis', u'golf', u'swimming']}

# Do projection first
# change _id to 'name'
# sort by 'name'
# only show result
users.aggregate([
	{'$project' : {'name': {'$toUpper' : '$_id'}, '_id':0}},
	{'$sort' : { 'name' : 1}}
])['result']

# Return user name by join month
users.aggregate([
	{'$project' : {'name': {'$toUpper' : '$_id'}, 
					'_id' : 0, 
					'month_joined' : {'$month':'$joined'}, 
					'joined' : '$joined'}},
	{'$sort' : {'month_joined' : 1}}
	])
# Return user name by month joined then user name
data = users.aggregate([
	{'$project' : {'name': {'$toUpper' : '$_id'}, 
					'_id' : 0, 
					'month_joined' : {'$month':'$joined'}, 
					'joined' : '$joined'}},
	{'$sort' : {'month_joined' : -1, 'name' : 1}}
	])

to_csv(data['result'])

data = users.aggregate([
	{'$project' : { 'month_joined' : {'$month' : '$joined'}}},
	{'$group' : { '_id' : {'month_joined' : '$month_joined'}, 
					'number' : {'$sum' : 1}}}
	])
to_csv(data['result'])

# Unwind operator
users.aggregate([
	{ '$unwind' : '$likes'},
	{'$match' : {'_id' : 'jane'}}
	])

# Return 5 most common 'likes'
users.aggregate([
	{ '$unwind' : '$likes'},
	{ '$group' : {'_id' : '$likes', 'number' : { '$sum' : 1}}},
	{ '$sort' : {'number': -1 }},
	{ '$limit' : 5}
	])
