"""setup.py

Distribution info file
"""
# Author: Dongjin Yoon <djyoon0223@gmail.com>


import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="analysis-tools",
    version="0.2.1",
    author="Dongjin Yoon",
    author_email="djyoon0223@gmail.com",
    description="Analysis tools for machine learning projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/djy-git/analysis-tools",
    install_requires=[
        'dask[complete]==2022.10.2',
        'dtreeviz==1.4.0',
        'joblib==1.2.0',
        'matplotlib==3.6.2',
        'missingno==0.5.1',
        'numba==0.56.4',
        'numpy==1.23.4',
        'pandas==1.5.1',
        'scikit-learn==1.1.3',
        'seaborn==0.12.1',
        'switch==1.1.0',
        'tabulate==0.9.0',
        'tqdm==4.64.1',
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
)
