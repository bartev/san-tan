"""
Multi Value Options
Tuples as Multi Value Options

https://click.palletsprojects.com/en/5.x/options/
"""
import click
import os
import sys

@click.command()
@click.option('--pos', nargs=2, type=float)
def findme(pos):
    """usage:
    ⇒ findme --pos 1.234 4.567
    1.234 / 4.567
    """
    click.echo(f'{pos[0]} / {pos[1]}')


@click.command()
@click.option('--item', type=(str, int))
def putitem(item):
    """usage:
    ⇒ putitem --item age 10
    name=age id=10
    """
    click.echo(f'name={item[0]} id={item[1]}')


@click.command()
@click.option('--item', nargs=2, type=click.Tuple([str, int]))
def putitem2(item):
    """usage:
    ⇒ putitem --item age 10
    name=age id=10
    """
    click.echo(f'name={item[0]} id={item[1]}')


@click.command()
@click.option('--message', '-m', multiple=True)
def commit(message):
    """usage:
    ⇒ commit -m foo -m bar
    foo
    bar
    """
    click.echo('\n'.join(message))


@click.command()
@click.option('-v', '--verbose', count=True)
def logme(verbose):
    """usage:
     logme -vvvv
    Verbosity: 4
    """
    click.echo(f'Verbosity: {verbose}')


@click.command()
@click.option('--shout/--no-shout', default=False)
def info2(shout):
    rv = sys.platform
    if shout:
        rv = f'{rv.upper()}!!!!1111'
    click.echo(rv)


@click.command()
@click.option('--shout', is_flag=True)
def info3(shout):
    rv = sys.platform
    if shout:
        rv = f'{rv.upper()}!!!!1111'
    click.echo(rv)

# Feature Switches


@click.command()
@click.option('--upper', 'transformation', flag_value='upper', default=True)
@click.option('--lower', 'transformation', flag_value='lower')
def trans(transformation):
    click.echo(getattr(sys.platform, transformation)())

# Choice Options


@click.command()
@click.option('--hash-type', type=click.Choice(['md5', 'sha1']))
def digest(hash_type):
    click.echo(hash_type)

# Prompting


@click.command()
@click.option('--name', prompt=True)
def hello_name(name):
    click.echo(f'Hello, {str.capitalize(name)}')
    click.echo(f'{name.capitalize()}, Hello')

# Password prompts

def rot13(password):
    """implement rotate by 13 ceasar cipher"""
    import codecs
    return codecs.encode(password, 'rot-13')


@click.command()
@click.option('--password', prompt=True, hide_input=True,
              confirmation_prompt=True)
def encrypt(password):
    click.echo(f'Encrypting password to {rot13(password)}')

@click.command()
@click.password_option()
def encrypt2(password):
    import codecs

    click.echo(f'Encrypting password 2 to {rot13(password)}')


# Dynamic defaults for prompts

@click.command()
@click.option('--username', prompt=True,
        default=lambda: os.environ.get('USER', ''))
def hello_un(username):
    print(f'Hello, {username}')


# Callbacks and eager options

def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo('Version 1.0')
    ctx.exit()

@click.command()
@click.option('--version', is_flag=True, callback=print_version, expose_value=False, is_eager=True)
def get_version():
    click.echo('Hello world')

# Yes parameters

def abort_if_false(ctx, param, value):
    if not value:
        ctx.abort()

@click.command()
@click.option('--yes', is_flag=True, callback=abort_if_false,
    expose_value=False,
    prompt='Are you sure you want to drop the db?')
def dropdb1():
    click.echo('dropped all tables')


