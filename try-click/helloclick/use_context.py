import click


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def cli(ctx, debug):
    # ensure ctx.obj exists and is a dict (in case `cli()` is
    # called outside of `if` block below)
    ctx.ensure_object(dict)

    ctx.obj['DEBUG'] = debug

@cli.command()
@click.pass_context
def sync(ctx):
    click.echo(f"Debug is {ctx.obj['DEBUG'] and 'on' or 'off'}")

if __name__ == "__main__":
    cli(obj={})