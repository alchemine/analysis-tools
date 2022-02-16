### Common parameters
RANDOM_STATE = 42


### Plot parameters
PLOT         = True
FIGSIZE_UNIT = 10
get_figsize  = lambda x, y: (int(FIGSIZE_UNIT * x), int(FIGSIZE_UNIT * y))
BINS         = 50


### Model selection
TEST_SIZE    = 0.2


### PATH(ROOT should be changed before using)
from os.path import join, dirname
class PATH:
    ROOT   = dirname(dirname(dirname(__file__)))
    INPUT  = join(ROOT, 'input')
    OUTPUT = join(ROOT, 'output')
    RESULT = join(ROOT, 'result')
    # TRAIN  = join(INPUT, 'train')
    # TEST   = join(INPUT, 'test')
    # CKPT   = join(ROOT, 'ckpt')
    # LOG    = join(ROOT, 'log')
