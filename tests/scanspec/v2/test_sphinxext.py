"""Tests for the `.. example_spec::` directive (scanspec.v2.sphinxext).

1.x's sphinxext.py has no dedicated test file either -- it's only ever
exercised by `tox -e docs` actually building real `.. example_spec::``
blocks found in the docs/docstrings. There's no v2 docs content yet (the
docs rewrite is explicitly deferred, PRD §12), so without a test here this
port would be completely unverified. Builds a real, minimal Sphinx project
from scratch rather than a full `tox -e docs` run -- the actual mechanism
this directive relies on (matplotlib's PlotDirective machinery), not a
mock of it.
"""

from __future__ import annotations

from pathlib import Path

from sphinx.application import Sphinx


def _build(tmp_path: Path, rst_body: str) -> Path:
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    (src / "conf.py").write_text(
        "extensions = [\n"
        '    "matplotlib.sphinxext.plot_directive",\n'
        '    "scanspec.v2.sphinxext",\n'
        "]\n"
    )
    (src / "index.rst").write_text(rst_body)

    app = Sphinx(
        srcdir=str(src),
        confdir=str(src),
        outdir=str(out),
        doctreedir=str(tmp_path / "doctrees"),
        buildername="html",
        status=None,
        warning=None,
    )
    app.build()
    assert app.statuscode == 0
    return out


def test_example_spec_directive_builds_a_figure(tmp_path: Path):
    out = _build(
        tmp_path,
        "Test\n"
        "====\n"
        "\n"
        ".. example_spec::\n"
        "\n"
        "    from scanspec.v2.specs import Linspace\n"
        "\n"
        '    spec = Linspace("x", 0, 1, 5)\n',
    )
    generated = list((out / "_images").glob("*.png"))
    assert len(generated) == 1
    assert generated[0].stat().st_size > 0


def test_example_spec_directive_works_with_a_multi_stream_spec(tmp_path: Path):
    """The auto-compile path (plot_scan(spec), no .compile() in the

    generated code) has to work for a real Acquire tree, not just a bare
    motion spec.
    """
    out = _build(
        tmp_path,
        "Test\n"
        "====\n"
        "\n"
        ".. example_spec::\n"
        "\n"
        "    from scanspec.v2.core import DetectorGroup\n"
        "    from scanspec.v2.specs import Acquire, Linspace\n"
        "\n"
        "    spec = Acquire(\n"
        '        Linspace("x", 0, 1, 5),\n'
        "        detectors=[DetectorGroup(1, 1, 0.01, 0.001, ['d1'])],\n"
        "    )\n",
    )
    generated = list((out / "_images").glob("*.png"))
    assert len(generated) == 1
