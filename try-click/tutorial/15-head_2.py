#!/usr/bin/env python


# http://zetcode.com/python/click/

import click


@click.command()
@click.argument('file_name', type=click.Path(exists=True))
@click.argument('lines', default=-1, type=int)
def head(file_name, lines):
    with open(file_name, 'r') as f:
        counter = 0
        for line in f:
            print(line.strip())
            counter += 1

            if counter == lines:
                break


if __name__ == '__main__':
    head()