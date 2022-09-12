"""Configuration module

Commonly used constant parameters are defined in capital letters.
"""
# Author: Dongjin Yoon <djyoon0223@gmail.com>


# Common parameters
class PARAMS:
    """Global parameters

    Parameters
    ----------
    seed : int
        Random seed

    test_size : float
        Size of test set size when splitting dataset
    """
    seed      = 42
    test_size = 0.2

    @classmethod
    def get(cls, key, val):
        """Return member variable(key) if val is None else val

        Parameters
        ----------
        key : str
            Name of parameter

        val : any dtype
            Value of the parameter

        >>> assert PARAMS.get('seed', 1) == 1
        >>> assert PARAMS.get('seed', None) == PARAMS.seed
        >>> assert PLOT_PARAMS.get('figsize', (20, 10)) == (20, 10)
        >>> assert PLOT_PARAMS.get('alpha', {'alpha': 0.5}) == 0.5
        """
        if val is None:
            return getattr(cls, key)
        else:
            if isinstance(val, dict):
                return val[key] if key in val else getattr(cls, key)
            return val


# Plot parameters
class PLOT_PARAMS(PARAMS):
    """Global plot parameters

    Parameters
    ----------
    show_plot : bool
        Whether to show plot

    figsize : tuple
        Figure size

    bins : int
        Number of bins for histogram plot

    n_classes : int
        Number of classes to show for categorical features

    n_cols : int
        Number of columns for subplots figure

    alpha : float
        Transparency

    marker_size : float
        Marker size for scatter plot
    """
    show_plot   = True
    figsize     = (30, 10)
    bins        = 50
    n_classes   = 5
    n_cols      = 5
    alpha       = 0.3
    marker_size = 20
