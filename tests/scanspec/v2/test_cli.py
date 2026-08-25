"""Tests for `python -m scanspec.v2` (the `plot` command)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
from click.testing import CliRunner

from scanspec.v2 import cli


def test_plot_default_view_is_path_only():
    runner = CliRunner()
    with patch("scanspec.v2.plot.plt.show"):
        result = runner.invoke(cli.cli, ["plot", "Linspace(x, 0, 1, 5)"])
    assert result.exit_code == 0, result.output
    assert len(plt.gcf().axes) == 1  # path only -- no detectors, no timeline anyway


def test_plot_view_scan_shows_timeline_when_detectors_present():
    runner = CliRunner()
    spec = (
        "Acquire(Linspace(x, 0, 1, 5), "
        "detectors=[DetectorGroup(1, 1, 0.01, 0.001, ['d1'])])"
    )
    with patch("scanspec.v2.plot.plt.show"):
        result = runner.invoke(cli.cli, ["plot", spec, "--view", "scan"])
    assert result.exit_code == 0, result.output
    assert len(plt.gcf().axes) == 2  # path + timeline


def test_plot_view_timeline_only():
    runner = CliRunner()
    spec = (
        "Acquire(Linspace(x, 0, 1, 5), "
        "detectors=[DetectorGroup(1, 1, 0.01, 0.001, ['d1'])])"
    )
    with patch("scanspec.v2.plot.plt.show"):
        result = runner.invoke(cli.cli, ["plot", spec, "--view", "timeline"])
    assert result.exit_code == 0, result.output
    assert len(plt.gcf().axes) == 1
    assert plt.gcf().axes[0].get_xlabel() == "time (s)"


def test_plot_axis_name_convenience_avoids_quoting():
    """Bare lowercase letters eval to their own name as a string."""
    runner = CliRunner()
    with patch("scanspec.v2.plot.plt.show"):
        result = runner.invoke(cli.cli, ["plot", "Linspace(x, 0, 1, 5)"])
    assert result.exit_code == 0, result.output
    assert plt.gcf().axes[0].get_xlabel() == "x"


def test_plot_output_saves_file_without_showing(tmp_path: Path):
    runner = CliRunner()
    out = tmp_path / "out.png"
    with patch("scanspec.v2.plot.plt.show") as mock_show:
        result = runner.invoke(
            cli.cli,
            ["plot", "Linspace(x, 0, 1, 5)", "--output", str(out), "--theme", "dark"],
        )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.stat().st_size > 0
    mock_show.assert_not_called()


def test_plot_title_option():
    runner = CliRunner()
    with patch("scanspec.v2.plot.plt.show"):
        result = runner.invoke(
            cli.cli, ["plot", "Linspace(x, 0, 1, 5)", "--title", "My Scan"]
        )
    assert result.exit_code == 0, result.output
    assert plt.gcf().get_suptitle() == "My Scan"
