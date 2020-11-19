#!/usr/bin/env python


# http://zetcode.com/python/click/

import click


@click.command()
@click.option('-s', '--string')
def output(string):
    click.echo(string)


if __name__ == '__main__':
    output()
