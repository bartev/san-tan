import click

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo('I was invoked w/o subcommand')
    else:
        click.echo(f'I am about to invoke {ctx.invoked_subcommand}')


@cli.command()
def sync():
    click.echo('the subcommand')

if __name__ == "__main__":
    cli()