#!/usr/bin/env python


# http://zetcode.com/python/click/

import click
from operator import mul
from functools import reduce


@click.command()
@click.argument('vals', type=int, nargs=-1)
def process(vals):
    print(f'The sum is {sum(vals)}')
    print(f'The product is {reduce(mul, vals, 1)}')


if __name__ == '__main__':
    process()
