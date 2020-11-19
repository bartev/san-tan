#!/usr/bin/env python


# http://zetcode.com/python/click/

import click


@click.command()
@click.option('--word', '-w', multiple=True)
def words(word):
    click.echo('\n'.join(word))


if __name__ == '__main__':
    words()
