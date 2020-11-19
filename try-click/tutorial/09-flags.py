#!/usr/bin/env python


# http://zetcode.com/python/click/

import click


# NOTE the use of `secho` instead of `echo`

@click.command()
@click.option('--blue', '-b', is_flag=True, help='message in blue color')
def hello(blue):
    if blue:
        click.secho('Hello there', fg='blue', bold=True)
    else:
        click.secho('Hello there', fg='red', bold=True)


if __name__ == '__main__':
    hello()
