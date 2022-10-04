from setuptools import setup, find_packages

VERSION = '0.0.2'
DESCRIPTION = 'Post-processing toolkit for materials testing data.'

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="paramaterial",
    version=VERSION,
    author="Daniel Slater",
    author_email="danielgslater@gmail.com",
    description=DESCRIPTION,
    long_description=long_description,
    long_descriptioncontent__type="text/markdown",
    packages=find_packages(),
    install_requires=['matplotlib'],
    keywords=['python', 'first package'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)