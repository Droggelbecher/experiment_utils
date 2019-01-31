
def is_sequence(a):
    try:
        iter(a)
    except TypeError:
        return False
    return True

def is_record(a):
    import numpy as np
    try:
        return a.dtype.names is not None
    except AttributeError:
        # Not a numpy-array-like thing, so not a record
        return False

