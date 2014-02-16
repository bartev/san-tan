db.users.remove()
db.users.insert({name: 'smith'})
db.users.insert({name: 'jones'})
db.users.count()
db.users.find({name: 'jones'})
db.users.find({ "_id" : ObjectId("5242202bf25f85a3b097020f"), "name" : "smith" })

db.users.find({$or: [
	{name: 'smith'},
	{name: 'jones'}
	]})


db.users.update({name: 'smith'}, {$set: {country: 'Canada'}})
db.users.update({name: 'smith'}, {country: 'Canada'})
db.users.update({name: 'smith'}, {$unset: {country: 1}})


db.users.update( {name: "smith"},
	{ $set: {favorites: {
			cities: ["Chicago", "Cheyenne"],
			movies: ["Casablanca", "The Sting"]}}})

db.users.update( {name: "jones"},
{ $set: {favorites: {
	movies: ["Casablanca", "Rocky"]
}}})


db.users.find({'favorites.movies':'Rocky'})

db.users.update({'favorites.movies':'Casablanca'},
{$addToSet: {'favorites.movies': 'The Maltese Flacon'} },
false,
true)
db.users.update({'favorites.movies':'Casablanca'},
{$addToSet: {'favorites.movies': 'Rocky'} },
false,
true)
db.users.update({'favorites.movies':'Casablanca'},
{$push: {'favorites.movies': 'The Sting'} },
false,
true)

for(i = 0; i < 200000; i++) { db.numbers.save({num: i}); }


db.numbers.find({num: {'$gt': 20, '$lt': 24}})

db.numbers.find( {num: {"$gt": 199995 }} ).explain()

db.numbers.ensureIndex({num: 1})
db.numbers.getIndexes()


mongo zoe.mongohq.com:10074/app15780382 -u heroku -p5397588f56035655939f3323bbef72bb


db.metrics.find({event: 'Tutorial', step:{$exists: true}}, 
	{step:1, 'player:level': 1, 'account:age:seconds':1}).limit(100)