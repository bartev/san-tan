import click


@click.group()
def cli():
    pass


@click.command()
def initdb():
    click.echo('Initialized the db')


@click.command()
def dropdb():
    click.echo('dropped the db')


cli.add_command(initdb)
cli.add_command(dropdb)

if __name__ == "__main__":
    cli()
