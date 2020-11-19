#!/usr/bin/env python


# http://zetcode.com/python/click/

import click


@click.command()
@click.option('--n', type=int, default=1)
def dots(n):
    click.echo('.' * n)


if __name__ == '__main__':
    dots()
