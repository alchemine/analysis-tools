"""Plot utility module

Commonly used plot related utility functions or classes are defined here.
"""
# Author: Dongjin Yoon <djyoon0223@gmail.com>


from analysis_tools.common.config import *
from analysis_tools.common.env import *


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

    title : str
        Super title of the figure.

    title_options : dict
        Options for super title.

    tight_layout : bool
        Whether to use tight layout.

    Examples
    --------
    >>> from analysis_tools.common.util import FigProcessor
    >>> fig, ax = plt.subplots()
    >>> with FigProcessor(fig, title="Feature distribution"):
    ...     ax.plot(...)
    """
    def __init__(self, fig, save_dir, title=None, tight_layout=True, title_options={}):
        self.fig           = fig
        self.save_dir      = save_dir
        self.title         = title
        self.tight_layout  = tight_layout
        self.title_options = title_options
        self.show_plot     = PLOT_PARAMS.show_plot
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
            if self.title:
                self.fig.suptitle(self.title, **self.title_options)
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        if self.save_dir:
            idx = 1
            while True:
                path = join(self.save_dir, f"{self.title}_{idx}.png")
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
