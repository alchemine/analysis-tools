"""Environment module

Commonly used packages and default settings are defined here.
"""
# Author: Dongjin Yoon <djyoon0223@gmail.com>


# Internal packages
import sys
import os
from os.path import join, isdir, isfile, exists, basename, dirname, split, abspath
import shutil
from glob import glob
from argparse import ArgumentParser
from configparser import ConfigParser
from importlib import import_module
import datetime
import joblib
import json
import re
from itertools import product, combinations, permutations, starmap
from functools import reduce
from time import time, sleep
from collections import defaultdict
from copy import deepcopy as copy
from tqdm import tqdm, trange
import warnings
import contextlib
from dataclasses import dataclass
from IPython.display import display, Markdown
import subprocess
import inspect
from abc import ABCMeta, abstractmethod


# External packages
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from tabulate import tabulate
import numba as nb
from numba import njit, cuda
import dask
from dask import delayed, compute
import dask.distributed
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import missingno as msno
import dateutil
from switch import Switch


# Plot packages
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib import gridspec
import seaborn as sns


# Plot options
#   Use korean fonts
"""
$ sudo apt-get install fonts-nanum* fontconfig
$ sudo fc-cache -fv
$ sudo cp /usr/share/fonts/truetype/nanum/Nanum* /opt/conda/envs/caret/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/
$ rm -rf /root/.cache/matplotlib/*
"""
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.rc('font', family='NanumGothic')
# plt.rc('font', family='DejaVu Sans')
plt.rc('axes', unicode_minus=False)
plt.rc('font', size=20)
plt.rc('figure', titlesize=40, titleweight='bold')
plt.style.use('ggplot')



# Set options
np.set_printoptions(suppress=True, precision=6, edgeitems=20, linewidth=100, formatter={"float": lambda x: "{:.3f}".format(x)})
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)


# Warning
warnings.filterwarnings('ignore')
