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