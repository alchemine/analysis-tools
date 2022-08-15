"""Configuration module

Commonly used constant parameters are defined in capital letters.
"""

# Author: Dongjin Yoon <djyoon0223@gmail.com>


# Common parameters
class PARAMS:
    seed      = 42
    test_size = 0.2

    @classmethod
    def get(cls, key, val):
        """
        >>> PLOT_PARAMS.get('figsize', (20, 10))
        >>> PLOT_PARAMS.get('alpha', {'alpha': 0.5})
        """
        if val is None:
            return getattr(cls, key)
        else:
            if isinstance(val, dict):
                if key in val:
                    return val[key]
                else:
                    return getattr(cls, key)
            return val


# Plot parameters
class PLOT_PARAMS(PARAMS):
    show_plot      = True
    figsize        = (30, 10)
    bins           = 50
    n_classes      = 5
    n_cols         = 5
    n_subsets_step = 5
    alpha          = 0.3
    s              = 20  # size of point in scatterplot
