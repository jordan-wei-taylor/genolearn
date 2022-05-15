import setuptools

def read(path):
    with open(path, encoding = 'utf-8') as f:
        return f.read()


setuptools.setup(
    name="genolearn",
    version="0.0.1",
    author="Jordan Taylor",
    author_email="jt2006@bath.ac.uk",
    description="A machine learning toolkit for genome sequence data",
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    url="https://github.com/jordan-wei-taylor/genolearn",
    project_urls={
        "Bug Tracker": "https://github.com/jordan-wei-taylor/genolearn/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.8",
    license=read('LICENSE'), 
    install_requires=[
        'scipy>=1.8.0',
        'pandas>=1.4.1',
        'numpy>=1.22.3',
        'psutil>=5.9.0',
        'scikit-learn>=1.0.2'
    ]
)