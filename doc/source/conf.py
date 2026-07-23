# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "spires-postprocess"
copyright = "2026, Niklas Griessbaum"
author = "Niklas Griessbaum"

# Version from the installed dist (setuptools_scm at build time); fall back to
# reading git tags directly in a source checkout.
try:
    from importlib.metadata import version

    release = version("spires-postprocess")
except Exception:
    try:
        from setuptools_scm import get_version

        release = get_version(root="../..", relative_to=__file__)
    except Exception:
        release = "unknown"

version = release

# -- General configuration ---------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",     # NumPy/Google docstrings
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",  # cross-link to the rest of the family
    "myst_parser",             # Markdown
    "sphinx_markdown_tables",
]

templates_path = ["_templates"]
exclude_patterns = []

autosummary_generate = True

# torch and lama are heavy, lazily-used deps; mock them so autodoc need not
# install them to introspect the public API.
autodoc_mock_imports = ["torch", "lama"]

# Cross-link to sibling packages' hosted docs (subprojects under the parent
# `spires` RTD project). Enable/extend as each publishes an objects.inv.
intersphinx_mapping = {
    "spires_contract": ("https://spires-contract.readthedocs.io/en/latest/", None),
}

suppress_warnings = ["myst.xref_missing"]

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
