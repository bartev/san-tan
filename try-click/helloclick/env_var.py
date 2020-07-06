import click
import os

# Values from environmental variables

# greet works only when being called with `auto_envvar_prefix` defined
@click.command()
@click.option('--username')
def greet(username):
    un = os.environ.get('GREETER_USERNAME', '')
    click.echo(f'from env: {un}')
    click.echo(f'Hello, {username}')


# greet2 does not work as expected
@click.command()
@click.option('--username')
def greet2(username, auto_envvar_prefix='GREETER'):
    un = os.environ.get('GREETER_USERNAME', '')
    click.echo(f'from env: {un}')
    click.echo(f'Hello, {username}')

@click.command()
@click.option('--username', envvar='GREETER_USERNAME')
def greet3(username):
    un = os.environ.get('GREETER_USERNAME', '')
    click.echo(f'from env: {un}')
    click.echo(f'Hello, {username}')


@click.command()
@click.option('paths', '--path', envvar='PATH', multiple=True, type=click.Path())
def perform(paths):
    for path in paths:
        click.echo(path)



if __name__ == "__main__":
    greet(auto_envvar_prefix='GREETER')