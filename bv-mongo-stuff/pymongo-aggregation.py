from pymongo import MongoClient
db = MongoClient().aggregation_example
db.things.insert({"x": 1, "tags": ["dog", "cat"]})
db.things.insert({"x": 2, "tags": ["cat"]})
db.things.insert({"x": 2, "tags": ["mouse", "cat", "dog"]})
db.things.insert({"x": 3, "tags": []})



# Aggregate
# ?!? HOW DOES THIS WORK?
# http://api.mongodb.org/python/current/examples/aggregation.html
from bson.son import SON
db.things.aggregate([
         {"$unwind": "$tags"},
         {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
         {"$sort": SON([("count", -1), ("_id", -1)])}
     ])

# group
from bson.son import SON
from bson.code import Code
reducer = Code("""
                function(obj, prev){
                  prev.count++;
                }
                """)
results = db.things.group(key={"x":1}, condition={}, initial={"count": 0}, reduce=reducer)
for doc in results:
	print doc