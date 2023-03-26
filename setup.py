from setuptools import setup

VERSION = '0.0.10'
DESCRIPTION = 'Software toolkit for parameterising materials test data. ' \
              'Easily batch process experimental measurements to determine mechanical properties and material model ' \
              'parameters.'


def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    with open('./requirements.txt') as f:
        return f.read().splitlines()


setup(name="paramaterial", version=VERSION, author="Daniel Slater", author_email="danielgslater@gmail.com",
    description=DESCRIPTION, long_description=readme(), long_description_content_type="text/markdown",
    url="https://github.com/dan-slater/paramaterial", install_requires=requirements(),
      packages=['paramaterial'],
    keywords=['python', 'mechanical testing', 'materials testing', 'post-processing', 'stress-strain', 'flow stress',
              'yield stress', 'tensile test', 'strength', 'elastic modulus', 'strain', 'universal testing machine',
              'fitting constitutive model', 'material model', 'solid mechanics', 'materials science', 'engineering'],
    classifiers=["Development Status :: 3 - Alpha", "Intended Audience :: Education",
        "Programming Language :: Python :: 3", "Operating System :: OS Independent", ])
