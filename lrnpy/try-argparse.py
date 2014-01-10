#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('square', help='display a square of a number', type=int)
args = parser.parse_args()
print args.square ** 2




parser.add_argument('square', help='display a square of a number', type=int)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--verbosity", help="increase output verbosity")
args = parser.parse_args()
if args.verbosity:
    print "verbosity turned on"


