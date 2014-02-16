from pymongo import Connection
db = Connection().map_reduce_example
db.things.insert({"x":1, "tags":['dog', 'cat']})
db.things.insert({"x":2, "tags":['cat']})
db.things.insert({"x":3, "tags":['mouse', 'cat', 'dog']})
db.things.insert({"x":4, "tags":[]})

from bson.code import Code
map = Code('function () {'
			'	this.tags.forEach(function(z) {'
			'		emit(z, 1);'
			'	});'
			'}')

reduce = Code('function (key, values) {'
					'var total = 0;'
					'for (var i = 0; i < values.length; i++) {'
					'	total += values[i];'
					'}'
					'return total;'
				'}')

result = db.things.map_reduce(map, reduce, "myresults")
for doc in result.find():
    print doc

result = db.things.map_reduce(map, reduce, "myresults", full_response=True)
result