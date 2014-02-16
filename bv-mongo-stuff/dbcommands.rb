require 'rubygems'
require 'mongo'
$con = Mongo::Connection.new
$admin_db = $con['admin']
ld = $admin_db.command({"listDatabases" => 1})
ld.each do |l|
  puts l
end