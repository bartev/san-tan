#!/usr/bin/env python

import click


@click.command()
@click.option('--name', '-n', default='World', help='Who to greet')
def byebye(name):
    click.echo(f'hello {name}')


if __name__ == '__main__':
    byebye(None)
    print('done')
