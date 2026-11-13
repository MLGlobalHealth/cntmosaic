# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

sys.path.insert(0, str(Path("../../").resolve()))

# -- Project information -----------------------------------------------------
project = "Contact Mosaic"
copyright = "2024–2026, Shozen Dan"
author = "Shozen Dan"

try:
    release = _pkg_version("cntmosaic")
except PackageNotFoundError:
    release = "unknown"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_nb",  # notebooks; also provides the MyST markdown parser
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "arviz": ("https://python.arviz.org/en/stable", None),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["**/.ipynb_checkpoints"]

# -- Tutorials (myst-nb) ------------------------------------------------------
# The notebooks in tutorials/ are committed without outputs (nbstripout), so they
# are executed when the docs are built. "cache" executes a notebook even if the
# working copy has outputs, and reuses the results in local rebuilds.
nb_execution_mode = "cache"
# Long-running tutorials are shown without outputs.
nb_execution_excludepatterns = [
    "tutorials/04_Tutorial_Feature_Selection.ipynb",
]
nb_execution_timeout = 900  # seconds per cell
nb_execution_raise_on_error = True  # a failing tutorial fails the docs build
nb_execution_show_tb = True

# Generate anchors for headings h1-h3, so in-page links such as [text](#slug) resolve.
myst_heading_anchors = 3

# autodoc settings
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "Contact Mosaic documentation"
html_theme_options = {
    "repository_url": "https://github.com/MLGlobalHealth/cntmosaic",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs/source",
}
