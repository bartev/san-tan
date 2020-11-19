#!/usr/bin/env python


# http://zetcode.com/python/click/

import click


@click.command()
@click.option('--name', prompt='Your name', help='Provide your name')
def hello(name):
    click.echo(f'Hello, {name}')


if __name__ == '__main__':
    hello()
