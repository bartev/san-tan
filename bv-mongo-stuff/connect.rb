# 
#  connect.rb
#  bv-mongo-stuff
#  
#  Created by Bartev Vartanian on 2013-09-25.
#  Copyright 2013 Bartev Vartanian. All rights reserved.
# 

require 'rubygems'
require 'mongo'
$con = Mongo::Connection.new
$db = $con['tutorial']
$users = $db['users']

$users.drop()

smith = {"last_name" => "smith", "age" => 30}
jones = {"last_name" => "jones", "age" => 40}

smith_id = $users.insert(smith)
jones_id = $users.insert(jones)

p $users.find_one({"_id" => smith_id})
p $users.find_one({"_id" => jones_id})

p "first query"
p $users.find_one({"last_name" => "smith"})

p "second query"
cursor = $users.find({"age" => {"$gt" => 20}})
cursor.each do |doc|
  puts doc["last_name"]
end

p "Add Chicago"
$users.update({"last_name" => "smith"}, {"$set" => {"city" => "Chicago"}})
cursor = $users.find()
cursor.each do |doc|
  puts doc["last_name"], doc["age"], doc["city"]
end

p "Change to NY"
$users.update({"last_name" => "smith"}, {"$set" => {"city" => "New York"}}, {:multi => true})
cursor = $users.find()
cursor.each do |doc|
  puts doc["last_name"], doc["age"], doc["city"]
end

$users.remove({"age" => {"$gte" => 40}})

$admin_db = $con['admin']
p $admin_db.command({"listDatabases" => 1})