#!/usr/bin/env python

# print the contents of the MYDIR (or mydir if specified) directory

# http://zetcode.com/python/click/

import click
from pathlib import Path


@click.argument('mydir', envvar='MYDIR', type=click.Path(exists=True))
@click.command()
def dolist(mydir):
    for p in sorted(Path(mydir).iterdir()):
        click.echo(p.name)


if __name__ == '__main__':
    dolist()
