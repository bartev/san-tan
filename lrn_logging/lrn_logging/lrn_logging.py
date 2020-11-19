#!/usr/bin/env python

"""Main module."""


import logging

log = logging.getLogger('__name__')


def do_something():
    log.debug('doing something')
    log.info('Hello, world')
    print('print me')


if __name__ == '__main__':
    do_something()
