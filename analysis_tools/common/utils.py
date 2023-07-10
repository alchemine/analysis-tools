"""Utility module

Commonly used utility functions or classes are defined here.
"""
# Author: Dongjin Yoon <djyoon0223@gmail.com>


from analysis_tools.common.config import *
from analysis_tools.common.env import *
from analysis_tools.common.plot_utils import *
from analysis_tools.common.timer import *


# Lambda functions
tprint  = lambda dic: print(tabulate(dic, headers='keys', tablefmt='psql'))  # print with fancy 'psql' format
vars_   = lambda obj: {k: v for k, v in vars(obj).items() if not k.startswith('__')}
ls_all  = lambda path: [path for path in glob(f"{path}/*")]
ls_dir  = lambda path: [path for path in glob(f"{path}/*") if isdir(path)]
ls_file = lambda path: [path for path in glob(f"{path}/*") if isfile(path)]
farray  = lambda shape, val=None: np.full(shape, val, dtype='float32')
iarray  = lambda shape, val=None: np.full(shape, val, dtype='int32')
str2dt = lambda s: datetime.datetime.strptime(s, "%Y-%m-%d")
dt2str = lambda dt: dt.strftime("%Y-%m-%d")

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
def yaml2dict(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
    if pd.api.types.is_numeric_dtype(data_f):
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
        return True
    except:
        return False


class MetaSingleton(type):
    """Superclass for singleton class
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
