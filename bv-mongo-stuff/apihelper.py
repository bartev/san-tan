#!/usr/bin/env python3


# 
#  apihelper.py
#  bv-mongo-stuff
#  
#  Created by Bartev Vartanian on 2013-09-27.
#  Copyright 2013 Bartev Vartanian. All rights reserved.
# 

def info(object, spacing=10, collapse=1):
    """Print methods and doc string
    
    Takes module, class, list, dictionary, or string."""
    methodList = [method for method in dir(object) if callable(getattr(object, method))]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s:s)
    print ("\n".join(["%s %s" %
                          (method.ljust(spacing),
                           processFunc(str(getattr(object, method).__doc__)))
                         for method in methodList]))

if __name__ == '__main__':
    print (info.__doc__)
