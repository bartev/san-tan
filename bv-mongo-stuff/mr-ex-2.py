from pymongo import Connection
db = Connection().map_reduce_example
db.custs.insert({"cust_id":'A123', "amount": 500, 'status': 'A'}) 
db.custs.insert({"cust_id":'A123', "amount": 250, 'status': 'A'})
db.custs.insert({"cust_id":'B212', "amount": 200, 'status': 'A'}) 
db.custs.insert({"cust_id":'A123', "amount": 300, 'status': 'D'})
	
from bson.code import Code
map = Code('function() {emit(this.cust_id, this.amount);}')
reduce = Code('function(key, values) {return Array.sum(values)}')
result = db.custs.map_reduce(map, reduce, query={'status': 'A'}, out='order_totals')
for doc in result.find(): print doc
