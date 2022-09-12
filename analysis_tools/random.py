"""Handling randomness analysis tools

Randomness controlling functions or classes are defined here.
"""
# Author: Dongjin Yoon <djyoon0223@gmail.com>


from analysis_tools.common import *


# Set random seed
def set_random_seed(seed=None):
    """
    Set random seed for reproducibility without using TensorFlow, PyTorch

    Parameters
    ----------
    seed : int
        Random seed
    """
    import numpy as np
    import random

    seed = PARAMS.get('seed', seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
def set_random_seed_tf(seed=None):
    """
    Set random seed for reproducibility on TensorFlow

    Parameters
    ----------
    seed : int
        Random seed
    """
    import tensorflow as tf

    seed = PARAMS.get('seed', seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.keras.utils.set_random_seed(seed)  # random, numpy, tensorflow
    tf.config.experimental.enable_op_determinism()
def set_random_seed_torch(seed=None):
    """
    Set random seed for reproducibility on PyTorch

    Parameters
    ----------
    seed : int
        Random seed
    """
    import numpy as np
    import random
    import torch

    seed = PARAMS.get('seed', seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
