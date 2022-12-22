import sys
import datetime
from os.path import abspath

for path in ['..']:
    sys.path.insert(0, abspath(path))
    print(abspath(path))

numpydoc_show_class_members = False 
doctest_global_setup = "import pluginmanager"
autodoc_default_flags = ['members']
autosummary_generate = True

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#sys.path.insert(0, os.path.abspath('.'))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
    'numpydoc',
    'sphinx_click.ext'
]


autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = ['.rst', '.md']

# The encoding of source files.
#source_encoding = 'utf-8-sig'


project = "GenoLearn"
author = "GenoLearn "
copyright = f"{datetime.datetime.now().year}, {author}"
version = release = "0.0.8"

todo_include_todos = False

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

html_logo = '_static/logo.png'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "collapse_navigation" : False,
    'logo_only' : True,
    'body_max_width': '100%',
    'style_nav_header_background' : "#fcfcfc"
}

html_show_sourcelink = False
html_use_index = True

html_static_path = ['_static']

html_js_files = [
    'js/custom.js'
]

html_style = 'css/custom.css'

numfig = True
numfig_secnum_depth = 2

html_favicon = 'favicon.png'

pngmath_latex_preamble=r'\usepackage[active]{preview}' # + other custom stuff for inline math, such as non-default math fonts etc.
pngmath_use_preview=True
