from analysis_tools.common.config import *
from analysis_tools.common.env import *


def print_guide():
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


### lambda functions
tprint  = lambda dic: print(tabulate(dic, headers='keys', tablefmt='psql'))  # print 'dic' with fancy 'psql' form
ls_all  = lambda path: [path for path in glob(f"{path}/*")]
ls_dir  = lambda path: [path for path in glob(f"{path}/*") if isdir(path)]
ls_file = lambda path: [path for path in glob(f"{path}/*") if isfile(path)]
figsize = lambda x, y: (int(FIGSIZE_UNIT * x), int(FIGSIZE_UNIT * y))


@dataclass
class Timer(ContextDecorator):
    name: str = ''
    def __enter__(self):
        self.start_time = time()
        return self
    def __exit__(self, *exc):
        elapsed_time = time() - self.start_time
        print(f"* {self.name}: {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)")
        return False


### Figure processor
class FigProcessor(ContextDecorator):
    def __init__(self, fig, dir_path, plot=True, suptitle='', suptitle_options={}, tight_layout=True):
        self.fig              = fig
        self.dir_path         = dir_path
        self.plot             = plot
        self.suptitle         = suptitle
        self.suptitle_options = suptitle_options
        self.tight_layout     = tight_layout
    def __enter__(self):
        pass
    def __exit__(self, *exc):
        if self.tight_layout:
            self.fig.suptitle(self.suptitle, **self.suptitle_options)
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        self.fig.savefig(join(self.dir_path, f"{self.suptitle}.png"))
        if self.plot:
            plt.show()
        plt.close(self.fig)
