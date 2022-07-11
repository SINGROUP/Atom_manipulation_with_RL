from setuptools import setup, find_packages
from os.path import join, dirname

requirementstxt = join(dirname(__file__), "requirements.txt")
requirements = [ line.strip() for line in open(requirementstxt, "r") if line.strip() ]

setup(
    name='AMRL',
    version='0.0.0',
    packages=find_packages(),
    url="https://atom-manipulation-with-rl.readthedocs.io/en/latest/",
    install_requires=requirements,

    # in order to make the documentation compile on readthedocs
    # we need to make pywin32 an 'extra' package since rtd runs on linux
    # thus to pip install and include pywin32 in requirements
    # we have to run ´pip install .[run]´
    extras_require={
        'run': [
            'pywin32'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ])
