"""Utility module

Commonly used utility functions or classes are defined here.
"""

# Author: Dongjin Yoon <djyoon0223@gmail.com>


from analysis_tools.common.config import *
from analysis_tools.common.env import *
from analysis_tools.common.plot_utils import *


def print_guide():
    """Print machine learning project guide for the user.
    """
    print(f"┌{' MACHINE LEARNING PROJECT GUIDE ':─^50}┐")
    print(f"│{' 1. Load data':<50}│")
    print(f"│{'    1.1 Define target':<50}│")
    print(f"│{' 2. Separate training, validation, test data':<50}│")
    print(f"│{' 3. Exploratory Data Analysis(EDA)':<50}│")
    print(f"│{'    3.1 Missing value':<50}│")
    print(f"│{'    3.2 Copy data':<50}│")
    print(f"│{'    3.3 Explore features':<50}│")
    print(f"│{'    3.4 Pair plot':<50}│")
    print(f"│{' 4. Preprocessing':<50}│")
    print(f"│{'    4.1 Split X, y':<50}│")
    print(f"│{'    4.2 Imputing':<50}│")
    print(f"│{'    4.3 Detailed preprocessing(feedback loop)':<50}│")
    print(f"│{' 5. Model selection':<50}│")
    print(f"│{' 6. Model tuning':<50}│")
    print(f"│{' 7. Evaluate the model on test data':<50}│")
    print(f"└{'─' * 50}┘ \n\n")


# Lambda functions
tprint  = lambda dic: print(tabulate(dic, headers='keys', tablefmt='psql'))  # print with fancy 'psql' format
ls_all  = lambda path: [path for path in glob(f"{path}/*")]
ls_dir  = lambda path: [path for path in glob(f"{path}/*") if isdir(path)]
ls_file = lambda path: [path for path in glob(f"{path}/*") if isfile(path)]
lmap    = lambda fn, arr: list(map(fn, arr))


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
