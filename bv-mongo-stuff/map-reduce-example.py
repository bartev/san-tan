import pymongo
import datetime
dt = datetime.datetime


# Get a db called test
client = MongoClient()
db = client.test
orders = db.orders

item = {
     'cust_id': "abc123",
     'ord_date': dt(2012, 10, 4),
     'status': 'A',
     'price': 25,
     'items': [ { 'sku': "mmm", 'qty': 5, 'price': 2.5 },
              { 'sku': "nnn", 'qty': 5, 'price': 2.5 } ]
}
item2 = {
     'cust_id': "xyz098",
     'ord_date': dt(2012, 10, 4),
     'status': 'A',
     'price': 15,
     'items': [ { 'sku': "mmm", 'qty': 5, 'price': 1.5 },
              { 'sku': "nnn", 'qty': 10, 'price': 0.75 } ]
}
item3 = {
     'cust_id': "abc123",
     'ord_date': dt(2012, 10, 4),
     'status': 'A',
     'price': 15,
     'items': [ { 'sku': "mmm", 'qty': 5, 'price': 1.5 },
              { 'sku': "nnn", 'qty': 10, 'price': 0.75 } ]
}
orders.insert(item3)


# Return total price per customer
# 1. define map function
from bson.code import Code
mapper = Code("""
	function() {
		emit(this.cust_id, this.price)
	}
	""")
reducer = Code("""
	function(keyCustId, valuesPrices) {
		return Array.sum(valuesPrices)
	}
	""")
results = orders.map_reduce(mapper, reducer, 'myresults')
results = orders.map_reduce(mapper, reducer, {'out': 'myresults'})

for doc in results.find(): print doc

# New example

from pymongo import Connection
db = Connection().map_reduce_example
db.things.insert({"x": 1, "tags": ["dog", "cat"]})
db.things.insert({"x": 2, "tags": ["cat"]})
db.things.insert({"x": 3, "tags": ["mouse", "cat", "dog"]})
db.things.insert({"x": 4, "tags": []})

from bson.code import Code
map = Code("""function() {
		this.tags.forEach(function(z) {
				emit(z, 1);
			})
		}
		""")
reduce = Code("""function(key, values) {
		var total = 0;
		for (var i = 0; i < values.length; i++) {
			total += values[i];
		}
		return total;
	}
	""")

# Simple map_reduce, send output to collection 'myresults'
result = db.things.map_reduce(map, reduce, null)
result = db.things.map_reduce(map, reduce, 'myresults')
# full response to map_reduce operation
result = db.things.map_reduce(map, reduce, 'myresults', full_response = True)
# write results to collection 'myresults' in the db 'outdb'
result = db.things.map_reduce(map, reduce, out = SON([('replace', 'myresults'), ('db', 'outdb')]))
for doc in result.find(): print doc