"""Utility module

Commonly used utility functions or classes are defined here.
"""

# Author: Dongjin Yoon <djyoon0223@gmail.com>


from analysis_tools.common.config import *
from analysis_tools.common.env import *


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


# Figure
class FigProcessor(contextlib.ContextDecorator):
    """Context manager for processing figure.

    Plot the figure and save it to the specified path.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to be processed.

    save_dir : str
        Directory path to save the figure.

    show_plot : bool
        Whether to show the figure.

    suptitle : str
        Super title of the figure.

    suptitle_options : dict
        Options for super title.

    tight_layout : bool
        Whether to use tight layout.

    Examples
    --------
    >>> from analysis_tools.common.util import FigProcessor
    >>> fig, ax = plt.subplots()
    >>> with FigProcessor(fig, suptitle="Feature distribution"):
    ...     ax.plot(...)
    """
    def __init__(self, fig, save_dir, show_plot=None, suptitle=None, suptitle_options={}, tight_layout=True):
        self.fig              = fig
        self.save_dir         = save_dir
        self.show_plot        = PLOT_PARAMS.get('show_plot', show_plot)
        self.suptitle         = suptitle
        self.suptitle_options = suptitle_options
        self.tight_layout     = tight_layout
    def __enter__(self):
        pass
    def __exit__(self, *exc):
        """Save and plot the figure.

        Parameters
        ----------
        exc : tuple
            Exception information.(dummy)
        """
        if self.tight_layout:
            if self.fig.suptitle:
                self.fig.suptitle(self.suptitle, **self.suptitle_options)
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        if self.save_dir:
            idx = 1
            while True:
                path = join(self.save_dir, f"{self.suptitle}_{idx}.png")
                if not exists(path):
                    break
                idx += 1
            self.fig.savefig(path)
        if self.show_plot:
            plt.show()
        plt.close(self.fig)

class SeabornFig2Grid:
    # https://stackoverflow.com/a/47664533
    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


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
    if is_numeric_dtype(data_f) or is_datetime64_ns_dtype(data_f):
        return 'num'
    else:
        return 'cat'
def is_datetime_format(s):
    """Check if the input string is datetime format or not

    Parameters
    ----------
    s : str
        String to be checked

    Returns
    ----------
    Whether the input string is datetime format or not
    """
    try:
        dateutil.parser.parse(s)
        return True
    except ValueError:
        return False
