from setuptools import setup, find_packages

VERSION = '0.0.4'
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
    long_description_content_type="text/markdown",
    url="https://github.com/dan-slater/paramaterial",
    packages=find_packages(),
    keywords=['python', 'mechanical testing', 'materials testing', 'post-processing', 'stress-strain', 'flow stress',
              'yield stress', 'tensile test', 'strength', 'elastic modulus', 'strain', 'universal testing machine',
              'fitting constitutive model', 'material model', 'solid mechanics', 'materials science', 'engineering'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
