"""Utility module

Commonly used utility functions or classes are defined here.
"""
# Author: Dongjin Yoon <djyoon0223@gmail.com>


from analysis_tools.common.config import *
from analysis_tools.common.env import *
from analysis_tools.common.plot_utils import *


# Lambda functions
tprint  = lambda dic: print(tabulate(dic, headers='keys', tablefmt='psql'))  # print with fancy 'psql' format
vars_   = lambda obj: {k: v for k, v in vars(obj).items() if not k.startswith('__')}
ls_all  = lambda path: [path for path in glob(f"{path}/*")]
ls_dir  = lambda path: [path for path in glob(f"{path}/*") if isdir(path)]
ls_file = lambda path: [path for path in glob(f"{path}/*") if isfile(path)]
farray  = lambda shape, val=None: np.full(shape, val, dtype='float32')
iarray  = lambda shape, val=None: np.full(shape, val, dtype='int32')

COLORS  = [c['color'] for c in plt.rcParams['axes.prop_cycle']]  # CAUTION: len(COLORS) == 7


def lmap(fn, arr, scheduler=None):
    if scheduler is None:
        return list(map(fn, arr))
    else:
        tasks = [delayed(fn)(e) for e in arr]
        return list(compute(*tasks, scheduler=scheduler))

def lstarmap(fn, *arrs, scheduler=None):
    assert np.unqiue(list(map(len, arrs))) == 1, "All parameters should have same length."
    if scheduler is None:
        return list(starmap(fn, arrs))
    else:
        tasks = [delayed(fn)(*es) for es in zip(*arrs)]
        return list(compute(*tasks, scheduler=scheduler))


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
def ini2dict(path):
    config = ConfigParser()
    config.read(path)
    return dict(config._sections)
str2dt = lambda s: datetime.datetime.strptime(s, "%Y-%m-%d")
dt2str = lambda dt: dt.strftime("%Y-%m-%d")
def double2float(data):
    num_cols = data.select_dtypes('float64').columns
    data[num_cols] = data[num_cols].astype('float32')
    return data

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
        # Check numerical type
        data_f.astype('float32')
        return False
    except:
        pass

    try:
        pd.to_datetime(data_f)
        # dateutil.parser.parse(sample)
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
    def __init__(self, name=''):
        self.name = name
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
