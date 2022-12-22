"""
logger doc
"""

from   multiprocessing import Value, Lock

import logging
import sys
import time
import os

clear = '\033[2K\033[1G'
up    = f'{clear}\033[F'
space = '    '
logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = f'{clear}%(asctime)s %(message)s', datefmt = "[%d-%m-%Y %H:%M:%S]")
logging.getLogger().handlers[0].terminator = ''
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)
    
def msg(text, inline = False, delete = 0, indent = 0):
    """
    Verbose message to print.

    Args:
        text (str)    : Message to print to sys.stdout.
        inline (bool) : If True, prints over the current line.
        delete (int)  : Number of previous lines to delete before printing.
        indent (int)  : Number of indentations (4 spaces) to indent text message by.
    """
    text = f'{space * indent}{text}' + ('' if inline else '\n')
    for _ in range(delete):
        logging.info(up)
    logging.info(text)


def print_dict(text, d):
    """
    Prints contents of dictionary using the msg function.

    Args:
        d (dict) : Dictionary of parameters. 
    """
    from  genolearn.utils   import set_params
    set_params(d)
    m    = max(len(key) for key in d)
    sep  = f'\n{clear}{" " * 23}'
    end  = f'\n{clear}{" " * 22}'
    text = f'{text}{end}' + '{' + f'{sep}' + f'{sep}'.join([f'{key:{m}s} : {val}' for key, val in d.items()]) + f'{end}' + '}'
    msg(text)


def _eta(seconds):
    """
    Computes eta in [hours,] minutes, seconds.

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: [hours,] minutes, seconds
    """
    minutes, seconds = divmod(seconds, 60)
    hours  , minutes = divmod(minutes, 60)
    if hours:
        return f'{hours:.0f}h {minutes:02.0f}m {seconds:02.0f}s'
    return f'{minutes:02.0f}m {seconds:02.0f}s'

class Update():
    """
    Update class designed for verbose print outs in loops.
    
    Args:
        obj (object) : Object to iterate over.
        msg (str)    : Base string to print out during object iteration.
        final (str)  : Final string to print out once object has been iterated.
        eta (bool)   : If True, includes estimated eta to print outs.
    """
    def __init__(self, obj, msg = '', final = None, eta = False):
        self.obj   = iter(obj)
        self.msg   = msg
        self._msg  = ''
        self.num   = len(obj)
        self.i     = 0
        self.final = final
        self.eta   = eta
        self.time  = time.time()

    def compute_msg(self):
        per     = self.i / self.num
        msg     = f'{self.msg} {self.i:,d} of {self.num:,d} ({per:.2%})'
        if self.eta and (self.num > self.i > 1):
            delta = time.time() - self.time
            eta   = delta / self.i * (self.num - self.i)
            msg   = f'{msg} | eta {_eta(eta)}'
        self._msg = msg

    def __iter__(self):
        return self

    def __next__(self):
        final = (self.num == self.i)
        if final:
            if self.final:
                msg(self.final)
            raise StopIteration
        self.i += 1
        self.compute_msg()
        msg(self._msg, inline = True)
        return next(self.obj)

class Waiting():

    def __init__(self, during, after, file = None, inline = True, delete = 0):
        self.during  = during
        self.after   = after
        self.file    = file if file and ' ' in file else f'"{os.path.basename(file)}"' if file else ''
        self.inline  = inline
        self.delete  = delete

    @property
    def _during(self):
        return f'{self.during} {self.file}' if self.file else self.during
    
    @property
    def _after(self):
        return f'{self.after} {self.file}' if self.file else self.after

    def __enter__(self):
        msg(f'{self._during}', inline = self.inline)

    def __exit__(self, *args):
        msg(f'{self._after}', delete = self.delete)

class Executing(Waiting):

    def __init__(self, file, inline = False, delete = 0):
        super().__init__('executing', 'executed', file = file, inline = inline, delete = delete)

class Computing(Waiting):

    def __init__(self, file = None, inline = False, delete = 0):
        super().__init__('computing', 'computed', file = file, inline = inline, delete = delete)

class Writing(Waiting):

    def __init__(self, file = None, inline = False, delete = 0):
        super().__init__('writing', 'written', file = file, inline = inline, delete = delete)

class Counter():

    def __init__(self, message, verbose = 0, display = True):
        self.val      = Value('i', 0)
        self.lock     = Lock()
        self.verbose  = verbose
        self.message  = message
        self.display  = display
        
    def increment(self):
        with self.lock:
            self.val.value += 1
            if self.display: msg(self.message.format(self.val.value), inline = self.val.value % self.verbose if self.verbose else True)

    def value(self):
        with self.lock:
            return self.val.value