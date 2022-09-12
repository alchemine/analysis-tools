"""Utility module

Commonly used utility functions or classes are defined here.
"""

# Author: Dongjin Yoon <djyoon0223@gmail.com>


from analysis_tools.common.config import *
from analysis_tools.common.env import *
from analysis_tools.common.plot_utils import *


# Lambda functions
tprint  = lambda dic: print(tabulate(dic, headers='keys', tablefmt='psql'))  # print with fancy 'psql' format
ls_all  = lambda path: [path for path in glob(f"{path}/*")]
ls_dir  = lambda path: [path for path in glob(f"{path}/*") if isdir(path)]
ls_file = lambda path: [path for path in glob(f"{path}/*") if isfile(path)]
lmap    = lambda fn, arr: list(map(fn, arr))


# Converter
def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError(f'Invalid input: {s} (type: {type(s)})')


# Check dtype
def dtype(data_f):
    """Return 'num' if data type is number or datetime else 'cat'

    Parameters
    ----------
    data_f : array-like
        Input array

    Returns
    -------
    Data Type : str
        Data type should be 'num' or 'cat'
    """
    if is_numeric_dtype(data_f):
        return 'num'
    else:
        return 'cat'
def is_datetime_str(data_f):
    """Check if the input string is datetime format or not

    Parameters
    ----------
    data_f : array-like
        str dtype array

    Returns
    ----------
    Whether the input string is datetime format or not
    """
    try:
        sample = data_f.unique()[0]
        dateutil.parser.parse(sample)
        return True
    except:
        return False


@dataclass
class Timer(contextlib.ContextDecorator):
    """Context manager for timing the execution of a block of code.

    Parameters
    ----------
    name : str
        Name of the timer.

    Examples
    --------
    >>> from time import sleep
    >>> from analysis_tools.common.util import Timer
    >>> with Timer('Code1'):
    ...     sleep(1)
    ...
    * Code1: 1.00s (0.02m)
    """
    name: str = ''
    def __enter__(self):
        """Start timing the execution of a block of code.
        """
        self.start_time = time()
        return self
    def __exit__(self, *exc):
        """Stop timing the execution of a block of code.

        Parameters
        ----------
        exc : tuple
            Exception information.(dummy)
        """
        elapsed_time = time() - self.start_time
        print(f"* {self.name}: {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)")
        return False


class MetaSingleton(type):
    """Superclass for singleton class
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
