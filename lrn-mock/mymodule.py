#!/usr/bin/env python

# https://www.toptal.com/python/an-introduction-to-mocking-in-python

import os


def rm(filename):
    if os.path.isfile(filename):
        os.remove(filename)
