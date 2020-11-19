#!/usr/bin/env python


# http://zetcode.com/python/click/

import click


@click.group()
def cli():
    pass


# add command directly to the `cli` function
@cli.command(name='gen')
def generic():
    click.echo('Hello there')


# define `welcome` as a command, then add to `cli`
# @cli.command(name='wel')
@click.command('wel2')
def welcome():
    click.echo('Welcome')


cli.add_command(welcome)

if __name__ == '__main__':
    cli()
