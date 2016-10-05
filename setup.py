# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='agent-trainer',
    version='0.1.0',
    description='Agent trainer',
    long_description=readme,
    author='Pedro Lopes',
    url='https://github.com/lopespm',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    entry_points={
          'console_scripts': [
              'machinedriver = machinedriver.__main__:main'
          ]
      },
)

