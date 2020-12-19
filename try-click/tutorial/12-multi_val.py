#!/usr/bin/env python


# http://zetcode.com/python/click/

import click


@click.command()
@click.option('--data', required=True, type=(str, int))
def output(data):
    click.echo(f'name={data[0]} age={data[1]}')


if __name__ == '__main__':
    output()
