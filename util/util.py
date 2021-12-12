### util.py ###################################
# Commonly used functions, classes are defined in here
###############################################


from util.env import *
from util.config import *


### lambda functions
tprint     = lambda dic: print(tabulate(dic, headers='keys', tablefmt='psql'))  # print 'dic' with fancy 'psql' form
display_md = lambda msg: display(Markdown(msg))
list_all   = lambda path: [(join(path, name), name) for name in sorted(os.listdir(path))]
list_dirs  = lambda path: [(join(path, name), name) for name in sorted(os.listdir(path)) if isdir(join(path, name))]
list_files = lambda path: [(join(path, name), name) for name in sorted(os.listdir(path)) if isfile(join(path, name))]

def rmdir(path):
    if isdir(path):
        shutil.rmtree(path)


### PATH
class PATH:
    ROOT   = abspath(dirname(os.getcwd()))
    SRC    = join(ROOT, 'src')
    INPUT  = join(ROOT, 'input')
    OUTPUT = join(ROOT, 'output')
    TRAIN  = join(INPUT, 'train')
    TEST   = join(INPUT, 'test')
    CKPT   = join(SRC, 'ckpt')
    RESULT = join(ROOT, 'result')
