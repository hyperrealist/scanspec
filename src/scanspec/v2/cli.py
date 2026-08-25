"""Interface for ``python -m scanspec.v2``."""

import logging
import string

import click
from matplotlib.figure import Figure

# Need this so we can eval() below -- v2 splits motion specs (Acquire,
# Linspace, ...) from detector/trigger types (DetectorGroup,
# TriggerSequence, ...) across two modules, unlike 1.x's single specs.py.
from .core import *  # noqa
from .plot import ThemeName, plot_path, plot_scan, plot_timeline
from .specs import *  # noqa

_VIEWS = {
    "scan": plot_scan,
    "path": plot_path,
    "timeline": plot_timeline,
}


@click.group(invoke_without_command=True)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(
        ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], case_sensitive=False
    ),
)
@click.version_option(prog_name="scanspec", message="%(version)s")
@click.pass_context
def cli(ctx: click.Context, log_level: str):
    """scanspec.v2 command line interface."""
    level = getattr(logging, log_level.upper(), None)
    logging.basicConfig(format="%(levelname)s:%(message)s", level=level)

    # if no command is supplied, print the help message
    if ctx.invoked_subcommand is None:
        # We need to prove that cli has been converted to a command
        # by the click decorator to keep pyright happy.
        assert isinstance(cli, click.Command)
        click.echo(cli.get_help(ctx))


@cli.command()
@click.argument("spec")
@click.option(
    "--view",
    default="path",
    type=click.Choice(["scan", "path", "timeline"]),
    help="scan: path + timeline together. path: motion only. "
    "timeline: detector triggering only.",
)
@click.option(
    "--theme",
    default="light",
    type=click.Choice(["light", "dark"]),
    help="Colour theme.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Save to this file instead of showing an interactive window.",
)
@click.option("--title", default=None, help="Figure title.")
def plot(spec: str, view: str, theme: ThemeName, output: str | None, title: str | None):
    """Plot a scanspec."""
    axis_names = {c: c for c in string.ascii_lowercase}
    eval_spec = eval(spec, globals(), axis_names)
    plot_fn = _VIEWS[view]

    if output is not None:
        fig = Figure()
        plot_fn(eval_spec, fig=fig, theme=theme, title=title)
        fig.savefig(output)  # type: ignore
    else:
        plot_fn(eval_spec, theme=theme, title=title)
