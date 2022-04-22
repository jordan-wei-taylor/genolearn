import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="genolearn",
    version="0.0.2",
    author="Jordan Taylor",
    author_email="jt2006@bath.ac.uk",
    description="A machine learning toolkit for genome sequence data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jordan-wei-taylor/genolearn",
    project_urls={
        "Bug Tracker": "https://github.com/jordan-wei-taylor/genolearn/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.8",
)