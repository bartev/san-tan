import click

@click.command()
@click.option('--count', '-c', default=1, help='number of greetings')
@click.argument('name')
def hello(count, name):
    """this script prints 'Hello NAME' COUNT times."""
    for x in range(count):
        click.echo(f'Hello {name}')

if __name__ == "__main__":
    hello()