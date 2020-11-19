#!/usr/bin/env python


# http://zetcode.com/python/click/

import click


# NOTE the use of `secho` instead of `echo`

@click.command()
def colored():
    click.secho('Hello there', fg='blue', bold=True)


if __name__ == '__main__':
    colored()
