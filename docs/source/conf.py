# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html






extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx_book_theme",
    "nbsphinx",
]

nbsphinx_execute = "never"



# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DeepDive'
copyright = '2025, Andreas Møller and Jesper Madsen'
author = 'Andreas Møller and Jesper Madsen'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_logo = "../_static/img/logo.png"
html_static_path = ['_static']

html_title = "DeepDive Documentation"

html_theme_options = {
    "repository_url": "https://github.com/madsen-lab/DeepDive",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
}