"""Configuration module

Commonly used constant parameters are defined in capital letters.
"""

# Author: Dongjin Yoon <djyoon0223@gmail.com>


# Common parameters
class PARAMS:
    seed      = 42
    test_size = 0.2

    @classmethod
    def get(cls, val, name):
        return getattr(cls, name) if val is None else val


# Plot parameters
class PLOT_PARAMS(PARAMS):
    show_plot      = True
    figsize        = (30, 10)
    bins           = 50
    n_classes      = 5
    n_cols         = 5
    n_subsets_step = 5
