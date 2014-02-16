#!/usr/bin/env python3

# 
#  scrap.py
#  bv-tex-notes-2
#  
#  Created by Bartev Vartanian on 2013-10-02.
#  Copyright 2013 Bartev Vartanian. All rights reserved.
# 

import re

# regex for a valid Roman numeral
# check thousands place
pattern = '^M?M?M?$'
re.search(pattern, 'M')
re.search(pattern, 'MM')
re.search(pattern, 'MMM')
re.search(pattern, 'MMMM')
re.search(pattern, '')

# check hundreds place
pattern = '^M?M?M?(CM|CD|D?C?C?C?)$'
re.search(pattern, 'MCM')
re.search(pattern, 'MD')
re.search(pattern, 'MMMCCC')
re.search(pattern, 'MCMC')
re.search(pattern, '')

# use {n, m} notation
pattern = '^M{0,3}$'  # no space!
re.search(pattern, 'M')
re.search(pattern, 'MM')
re.search(pattern, 'MMM')
re.search(pattern, 'MMMM')
re.search(pattern, '')

# check for tens
pattern = '^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})$'  # no space!
re.search(pattern, 'MCMXL')
re.search(pattern, 'MCML')
re.search(pattern, 'MCMLX')
re.search(pattern, 'MCMLXXX')
re.search(pattern, 'MCMLXXXX')

# check for ones
pattern = '^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
re.search(pattern, 'MDLV')
re.search(pattern, 'MMDCLXVI')
re.search(pattern, 'MMMDCCCLXXXVIII')
re.search(pattern, 'I')


# verbose regex
# ignore whitespace
# allow comments
# python assumes compact regex
# use flag 're.VERBOSE' when searching
pattern = '''
^                   # beginning of string
M{0,3}              # thousands - 0 to 3 Ms
(CM|CD|D?C{0,3})    # hundreds - 900 (CM), 400 (CD), 0-300 (0 to 3 Cs),
                    #           or 500-800 (D, followed by 0 to 3 Cs)
(XC|XL|L?X{0,3})    # tens - 90 (XC), 40 (XL), 0-30 (0 to 3 Xs),
                    #       or 50-80 (L, followed by 0 to 3 Xs)
(IX|IV|V?I{0,3})    # ones - 9 (IX), 4 (IV), 0-3 (0 to3 Is),
                    #       or 5-8 (V, followed by 0 to 3 Is)
$                   # end of string
'''

re.search(pattern, 'M', re.VERBOSE)
re.search(pattern, 'MCMLXXXIX', re.VERBOSE)
re.search(pattern, 'MMMDCCCLXXXVIII', re.VERBOSE)
re.search(pattern, 'M')


# parsing an American phone number
phonePattern = re.compile(r'^(\d{3})-(\d{3})-(\d{4})$')
phonePattern.search('800-555-1212').groups()
phonePattern.search('800-555-1212-1234')
phonePattern.search('800-555-1212-1234').groups()

# allow for phone extension
phonePattern = re.compile(r'^(\d{3})-(\d{3})-(\d{4})-(\d+)$')
phonePattern.search('800-555-1212-1234').groups()
phonePattern.search('800 555 1212 1234')
phonePattern.search('800-555-1212')

# handle separators btw different parts of the number
phonePattern = re.compile(r'^(\d{3})\D+(\d{3})\D+(\d{4})\D+(\d+)$')
phonePattern.search('800 555 1212 1234').groups()
phonePattern.search('800-555-1212-1234').groups()
phonePattern.search('80055512121234')
phonePattern.search('800-555-1212')

# 0 or more separators & optional extension
phonePattern = re.compile(r'^(\d{3})\D*(\d{3})\D*(\d{4})\D*(\d*)$')
phonePattern.search('800 555 1212 1234').groups()
phonePattern.search('800-555-1212-1234').groups()
phonePattern.search('800.555.1212 x1234').groups()
phonePattern.search('80055512121234').groups()
phonePattern.search('(800)5551212 x1234').groups()
phonePattern.search('800-555-1212').groups()

# allow leading parentheses
phonePattern = re.compile(r'^\D*(\d{3})\D*(\d{3})\D*(\d{4})\D*(\d*)$')
phonePattern.search('800 555 1212 1234').groups()
phonePattern.search('800-555-1212-1234').groups()
phonePattern.search('800.555.1212 x1234').groups()
phonePattern.search('80055512121234').groups()
phonePattern.search('(800)5551212 x1234').groups()
phonePattern.search('800-555-1212').groups()
phonePattern.search('work 1-(800) 555.1212 #1234').groups()

# match anywhere in the string (still not perfect - see last example)
phonePattern = re.compile(r'(\d{3})\D*(\d{3})\D*(\d{4})\D*(\d*)$')
phonePattern.search('800 555 1212 1234').groups()
phonePattern.search('800-555-1212-1234').groups()
phonePattern.search('800.555.1212 x1234').groups()
phonePattern.search('80055512121234').groups()
phonePattern.search('(800)5551212 x1234').groups()
phonePattern.search('800-555-1212').groups()
phonePattern.search('work 1-(800) 555.1212 #1234').groups()
phonePattern.search('here is 333 my work number 893 if you want 3453').groups()

# write as verbose regex
phonePattern = re.compile(r'''
            # don't match beginning of string, number can start anywhere'
(\d{3})     # area code is 3 digits (e.g. '800')
\D*         # optional separator is any number of non-digits
(\d{3})     # trunk is 3 gigits (e.g. '555')
\D*         # optional separator
(\d{4})     # rest of number is 4 digits (e.g. '1212')
\D*         # optional separator
(\d*)       # extension is optional and can be any number of digits
$           # end of string
''', re.VERBOSE)
phonePattern.search('800 555 1212 1234').groups()
phonePattern.search('800-555-1212-1234').groups()
phonePattern.search('800.555.1212 x1234').groups()
phonePattern.search('80055512121234').groups()
phonePattern.search('(800)5551212 x1234').groups()
phonePattern.search('800-555-1212').groups()
phonePattern.search('work 1-(800) 555.1212 #1234').groups()
phonePattern.search('here is 333 my work number 893 if you want 3453').groups()
