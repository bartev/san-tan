"""
`foo` is the click group.
It is defined as the entry point in setup.py.

`foo` has a sub command, `say`
`say` is a click command, defined using `foo.command()`
Alternatively, it can be define using `click.command()`, and
then added to `foo` using  `foo.add_command(shutit)`
https://click.palletsprojects.com/en/5.x/quickstart/

"""

import click


class Config (object):
    def __init__(self):
        self.verbose = False


pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option('--verbose', is_flag=True)
@click.option('--home-directory', type=click.Path())
@pass_config
def foo(config, verbose, home_directory):
    config.verbose = verbose
    if home_directory is None:
        home_directory = '.'
    config.home_directory = home_directory


@foo.command()
@click.option('--string', default='World',
              help='This is the person who is greeted')
@click.option('--repeat', default=1,
              help='how many times to repeat')
@click.argument('out', type=click.File('w'), default='-',
                required=True)
@pass_config
def say(config, string, repeat, out):
    """This script greets you"""
    # click.echo (out)
    if config.verbose:
        click.echo('we are in verbose mode')

    click.echo(f'Home dir is {config.home_directory}')
    out.write('start here\n')
    for _ in range(repeat):
        print(f'Hello {string}')
        click.echo(f'Howdy, {string} doofus', file=out)


@click.command()
@click.option('--string', '-s', default='Bub',
              help='Who should shut up')
@pass_config
def shutit(config, string):
    """Tell you to shut it"""
    if config.verbose:
        click.echo('we are in verbose mode')

    print(f'Shut it, {string}!')


foo.add_command(shutit)
