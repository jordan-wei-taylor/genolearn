from setuptools import setup
from genolearn  import __version__
from pathlib    import Path

url = "https://github.com/jordan-wei-taylor/genolearn"

setup(
    name             = "genolearn",
    version          = __version__,
    author           = "Jordan Taylor",
    author_email     = "jt2006@bath.ac.uk",
    description      = "A machine learning toolkit for genome sequence data",
    long_description = (Path(__file__).parent / 'README.md').read_text(),
    long_description_content_type = 'text/markdown',
    license          = 'BSD-3-Clause',
    package          = ['genolearn'],
    py_modules       = [],
    url              = url,
    project_urls     = {"Bug Tracker": f"{url}/issues"},
    classifiers      = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires  = ">=3.10",
    install_requires = [
        'click>=8.1.3',        # cli
        'numpy>=1.22.3',       # core
        'pandas>=1.4.2',       # core
        'pathos>=0.3.0',       # parallelisation
        'psutil>=5.9.0',       # logging RAM
        'scikit-learn>=1.1.2', # core
        'scipy>=1.8.0',        # sparse arrays
    ],
    entry_points     = '''
        [console_scripts]
        genolearn=genolearn.cli:menu
        genolearn-clean=genolearn.cli:clean
        genolearn-setup=genolearn.cli:setup
    ''',
    
)