
from .calltree import Operation

def operation():
    def wrapper(f):
        return Operation(f)
    return wrapper


def demo():

    import numpy as np
    from .storage import DiskStorage
    from .session import Session

    @operation()
    def foo(a):
        print("actually computing: foo a=", a)
        return a + 1

    @operation()
    def bar(a):
        print("actually computing: bar a=", a)
        return a * 2

    # During `compute(c)` this will cache intermediate results
    # In next run it will only load the hash of `b` and `c` from disk.
    # Only in the final `.load_value()`, `c`s actual value is loaded from disk.
    a = np.arange(10000000)
    b = foo(a)
    c = bar(b)

    s = Session(storage=DiskStorage('_cache_test'))
    print('compute=',s.compute(c).load_value())
    print('compute=',s.compute(c).load_value())


def print_cachefile(filename):
    from pprint import pprint
    from .storage import DiskStorage

    pprint(DiskStorage.load_meta(filename))

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        demo()
    else:
        print_cachefile(sys.argv[1])
