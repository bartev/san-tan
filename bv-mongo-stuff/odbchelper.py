#!/usr/bin/env python

# 
#  odbchelper.py
#  bv-mongo-stuff
#  
#  Created by Bartev Vartanian on 2013-09-26.
#  Copyright 2013 Bartev Vartanian. All rights reserved.
# 
#  From http://www.diveintopython.net/getting_to_know_python/testing_modules.html

def buildConnectionString(params):
	"""Build a connection string
	Returns a string."""
	return ";".join(["%s=%s" % (k, v) for k, v in params.items()])

if __name__ == '__main__':
    myParams = {"server":"mpilgrim",
    "database":"master",
    "uid":"sa",
    "pwd":"secret"}
    print (buildConnectionString(myParams))
	