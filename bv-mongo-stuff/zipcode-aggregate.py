
# From command line
# mongoimport --collection zipcodes --file zips.json
# imported collection zipcodes into test

# mongo code
# > db.zipcodes.aggregate( {$group : {_id: '$state', totalPop : { $sum : '$pop' }}}, 
						 # {$match : {totalPop : {$gte : 10 * 1000 * 1000 }}}
						 # )

def to_csv(data, delim=",", output=sys.stdout):
    """docstring for to_csv"""
    if type(data) is dict: data = [data]  # wrap in list if it's a dict

    all_keys = set()
    if len(data):
        for row in data:
            all_keys = set.union(all_keys, set(row.keys()))
    ks = list(all_keys)
    ks.sort()

    writer = csv.writer(output, delimiter=delim)
    writer.writerow(ks)
    for row in data:
        vs = []
        for k in ks:
            vs.append(row.get(k, None))
        writer.writerow(vs)
    return output

# Get a db called test
client = MongoClient()
db = client.test

# Gives a dict object
db.zipcodes.aggregate(
	{'$group' : {'_id': '$state', 'totalPop' : {'$sum' : '$pop'}}}
	)
# >>> db.zipcodes.aggregate(
# ...     {'$group' : {'_id': '$state', 'totalPop' : {'$sum' : '$pop'}}}
# ...     )
# {u'ok': 1.0, u'result': [{u'_id': u'WV', u'totalPop': 1793477}, {u'_id': u'WA', u'totalPop': 4866692}, {u'_id': u'VA', u'totalPop': 6187358}, {u'_id': u'VT', u'totalPop': 562758}, {u'_id': u'UT', u'totalPop': 1722850}, {u'_id': u'TN', u'totalPop': 4876457}, {u'_id': u'SC', u'totalPop': 3486703}, {u'_id': u'RI', u'totalPop': 1003464}, {u'_id': u'PA', u'totalPop': 11881643}, {u'_id': u'OK', u'totalPop': 3145585}, {u'_id': u'OH', u'totalPop': 10847115}, {u'_id': u'ND', u'totalPop': 638800}, {u'_id': u'NY', u'totalPop': 17990455}, {u'_id': u'OR', u'totalPop': 2842321}, {u'_id': u'NM', u'totalPop': 1515069}, {u'_id': u'NH', u'totalPop': 1109252}, {u'_id': u'NE', u'totalPop': 1578385}, {u'_id': u'TX', u'totalPop': 16986510}, {u'_id': u'MO', u'totalPop': 5114343}, {u'_id': u'DC', u'totalPop': 606900}, {u'_id': u'MN', u'totalPop': 4375099}, {u'_id': u'LA', u'totalPop': 4219973}, {u'_id': u'WI', u'totalPop': 4891769}, {u'_id': u'ID', u'totalPop': 1006749}, {u'_id': u'KY', u'totalPop': 3685296}, {u'_id': u'WY', u'totalPop': 453588}, {u'_id': u'MI', u'totalPop': 9295297}, {u'_id': u'KS', u'totalPop': 2477574}, {u'_id': u'CO', u'totalPop': 3294394}, {u'_id': u'MS', u'totalPop': 2573216}, {u'_id': u'AZ', u'totalPop': 3665228}, {u'_id': u'NC', u'totalPop': 6628637}, {u'_id': u'IA', u'totalPop': 2776755}, {u'_id': u'MD', u'totalPop': 4781468}, {u'_id': u'ME', u'totalPop': 1227928}, {u'_id': u'IN', u'totalPop': 5544159}, {u'_id': u'MT', u'totalPop': 799065}, {u'_id': u'AL', u'totalPop': 4040587}, {u'_id': u'CT', u'totalPop': 3287116}, {u'_id': u'HI', u'totalPop': 1108229}, {u'_id': u'FL', u'totalPop': 12937926}, {u'_id': u'MA', u'totalPop': 6016425}, {u'_id': u'GA', u'totalPop': 6478216}, {u'_id': u'SD', u'totalPop': 696004}, {u'_id': u'AK', u'totalPop': 550043}, {u'_id': u'NJ', u'totalPop': 7730188}, {u'_id': u'IL', u'totalPop': 11430602}, {u'_id': u'AR', u'totalPop': 2350725}, {u'_id': u'NV', u'totalPop': 1201833}, {u'_id': u'DE', u'totalPop': 666168}, {u'_id': u'CA', u'totalPop': 29760021}]}

# Gives a dict object
# NOTE: unlike the mongo query above, put the group and match in a list
db.zipcodes.aggregate([
	{'$group' : {'_id': '$state', 'totalPop' : {'$sum' : '$pop'}}},
	{'$match' : {'totalPop' : {'$gte' : 10 * 1000 * 1000 }}}
	])

# >>> res = db.zipcodes.aggregate([
# ...     {'$group' : {'_id': '$state', 'totalPop' : {'$sum' : '$pop'}}},
# ...     {'$match' : {'totalPop' : {'$gte' : 10 * 1000 * 1000 }}}
# ...     ])
# >>> type(res)
# <type 'dict'>

# Equivalent SQL is
# SELECT state, SUM(pop) AS totalPop
#        FROM zipcodes
#        GROUP BY state
#        HAVING totalPop >= (10*1000*1000)


db.zipcodes.aggregate([
	{'$group': {'_id' : { 'state' : '$state', 'city' : '$city' },
				'pop' : { '$sum' : '$pop' }}}
	])

# results: [{u'_id': {u'city': u'MEKORYUK', u'state': u'AK'}, u'pop': 177}, ...]


db.zipcodes.aggregate([
	{'$group': {'_id' : { 'state' : '$state', 'city' : '$city' },
				'pop' : { '$sum' : '$pop' }}},
	{'$group': {'_id' : '$_id.state', 'avgCityPop' : { '$avg' : '$pop' }}}
	])
# results look like
# {u'ok': 1.0, u'result': [{u'_id': u'NH', u'avgCityPop': 5232.320754716981}, {u'_id': u'RI', u'avgCityPop': 18933.283018867925}, {u'_id': u'HI', u'avgCityPop': 15831.842857142858}, {u'_id': u'FL', u'avgCityPop': 26676.136082474226}, {u'_id': u'NV', u'avgCityPop': 18209.590909090908}, {u'_id': u'DE', u'avgCityPop': 14481.91304347826}, {u'_id': u'SC', u'avgCityPop': 11139.626198083068}, {u'_id': u'CT', u'avgCityPop': 14674.625}

# 2 steps in pipeline:
# 1: get city/state population
# 2: get avg city population by state

# Try one of my own. Get total population by state 2 ways
regx = re.compile('^A', re.IGNORECASE)
db.zipcodes.aggregate([
	{'$group': {'_id' : { 'state' : '$state'},
				'pop' : { '$sum' : '$pop' }}}
	])
db.zipcodes.aggregate([
	{'$group': {'_id' : { 'state' : '$state', 'city' : '$city' },
				'pop' : { '$sum' : '$pop' }}},
	{'$group': {'_id' : '$_id.state', 'sumCityPop' : { '$sum' : '$pop' }}}
	])
# YES. They both give the same result!

# doesn't work
# db.zipcodes.aggregate([
# 	{'$group': {'_id' : { 'state' : '$state'},
# 				'pop' : { '$sum' : '$pop' }}},
# 	{'$match': {'state' : 'CA'}}
# 	])


# Largest and smallest city by state
db.zipcodes.aggregate([{ '$group' : { '_id' : {'state' : '$state', 'city' : '$city'},
									'pop': {'$sum': '$pop' }}},
						{ '$sort' : {'pop': 1} },
						{'$group': {'_id': '$_id.state',
									'biggestCity': {'$last': '$_id.city'},
									'biggestPop': {'$last': '$pop'},
									'smallestCity': {'$first': '$_id.city'},
									'smallestPop': {'$first': '$pop'}}}
					])
# output
# {u'ok': 1.0, u'result': 
# [	{u'biggestPop': 176404, u'_id': u'RI', 
# 		u'biggestCity': u'CRANSTON', u'smallestCity': u'CLAYVILLE', u'smallestPop': 45}, 
# 	{u'biggestPop': 536759, u'_id': u'OH', 
# 		u'biggestCity': u'CLEVELAND', u'smallestCity': u'ISLE SAINT GEORG', u'smallestPop': 38}, 
# 	{u'biggestPop': 733081, u'_id': u'MD', 
# 		u'biggestCity': u'BALTIMORE', u'smallestCity': u'ANNAPOLIS JUNCTI', u'smallestPop': 32}, 
# 	{u'biggestPop': 106452, u'_id': u'NH', 
# 		u'bigges

# Now add a projection
regx = re.compile('^A', re.IGNORECASE)

db.zipcodes.aggregate([{ '$group' : { '_id' : {'state' : '$state', 'city' : '$city'},
									'pop': {'$sum': '$pop' }}},
						{ '$sort' : {'pop': 1} },
						{'$group': {'_id': '$_id.state',
									'biggestCity': {'$last': '$_id.city'},
									'biggestPop': {'$last': '$pop'},
									'smallestCity': {'$first': '$_id.city'},
									'smallestPop': {'$first': '$pop'}}},
						{'$project':
							{'_id': 0,
							'state': '$_id',		# rename _id field to 'state'
							'biggestCity': {'name': '$biggestCity', 'pop':'$biggestPop'},
							'smallestCity': {'name': '$smallestCity', 'pop':'$smallestPop'},
							}},
						{'$sort' : {'state':1}}, 
						{'$match' : {'state': regx}}
					])
						# {'$match' : {'state': 'CA'}}
 # output:
 # {u'ok': 1.0, u'result': 
 # [	{u'state': u'RI', 
 # 		u'biggestCity': {u'name': u'CRANSTON', u'pop': 176404}, 
 # 		u'smallestCity': {u'name': u'CLAYVILLE', u'pop': 45}}, 
 # 	{u'state': u'OH', 
 # 		u'biggestCity': {u'name': u'CLEVELAND', u'pop': 536759}, 
 # 		u'smallestCity': {u'name': u'ISLE SAINT GEORG', u'pop': 38}}, 
	# {u'state': u'MD', 
	# 	u'biggestCity': {u'name': u'BALTIMORE', u'pop': 733081}, 
	# 	u'smallestCity': {u'name': u'ANNAPOLIS JUNCTI', u'pop': 32}}, 
	# {u'state': u'NH', 
	# 	u'biggestCity': {u'name': u'MANCHESTER', u'pop': 106452}, 
	# 	u'smallestCity': {u'name': u'WEST NOTTINGHAM', u'pop': 27}}, 
	# {u'state': u'IN', 
	# 	u'biggestCity': {u'name': u'INDIANAPOLIS', u'pop': 348868}, 
	# 	u'smallestCity': {u'name': u'47559', u'pop': 23}}, 
	# {u'state': u'MA'