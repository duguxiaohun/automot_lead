"""Interpreter boundary for frozen launchers using torchrun module mode.

Keep checkpoint-bound student sources unchanged. Explicitly end torchrun's
options before the application module, so --run cannot abbreviate --run-path.
"""
import os
from pathlib import Path
import sys


def arguments(argv):
    argv=list(argv)
    if argv[:2]!=['-m','torch.distributed.run']:
        return argv
    for i in range(2,len(argv)):
        if argv[i]=='--':return argv
        if argv[i] in ('-m','--module'):
            if i+1==len(argv):raise ValueError('missing torchrun application module')
            if argv[i+1]=='--':return argv
            return argv[:i+1]+['--']+argv[i+1:]
    return argv  # Script launchers keep their own positional grammar.


def environment():
    return dict(PYTHON=str(Path(__file__).with_name('python_dispatch.sh')),
                AUTOMOT_REAL_PYTHON=sys.executable)


def main():
    os.execv(sys.executable,[sys.executable,*arguments(sys.argv[1:])])


if __name__=='__main__':main()
