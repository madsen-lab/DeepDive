from setuptools import setup, find_packages
import os

if sys.version_info.major != 3:
    raise RuntimeError("DeepDive requires Python 3")

with open("README.md", "r") as fh:
    long_description = fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'DeepDive'

# Setting up
setup(
    name="DeepDive",
    version=VERSION,
    author=['Andreas Fønss Møller','Jesper Grud Skat Madsen'],
    author_email=['andreasfm@bmb.sdu.dk', 'jgsm@imada.sdu.dk'],
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'torch'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)