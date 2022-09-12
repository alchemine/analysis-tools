"""Utility analysis tools

Utility functions or classes are defined here.
"""
# Author: Dongjin Yoon <djyoon0223@gmail.com>


from analysis_tools.common import *


def set_memory_growth():
    """Allocate only the GPU memory needed for runtime.
    """
    import tensorflow as tf

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def check_nan(data, name):
    """Print number of data and nan rows

    Parameters
    ----------
    data : pandas.DataFrame
        Input data
    name : str
        Name to identify data
    """
    print("* Data name:", name)
    print("  - Number of rows:", len(data))
    print("  - Number of nan rows:", sum(data.isna().sum(axis='columns') > 0))
