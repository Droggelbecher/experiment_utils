
try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from zlib import adler32 as _hash

def cache_hash(obj):
    """
    >>> cache_hash('foo') == cache_hash('f' + 'o'*2) == cache_hash( ''.join(['f', 'o', 'o'])) == cache_hash('foobar'[:3])
    True
    >>> cache_hash(2 + 3) == cache_hash(5) == cache_hash( 1000 // 200) == cache_hash( int(5.0)) == cache_hash(int("5"))
    True
    >>> cache_hash(4000) == cache_hash(2000*2) == cache_hash( 40000 // 10) == cache_hash( int(4000.0)) == cache_hash(int("4000"))
    True
    >>> cache_hash( () ) == cache_hash( (1,2,3)[3:3] ) == cache_hash(tuple(list()))
    True
    >>> cache_hash( (1,2,3)) == cache_hash(tuple([1,2,3]))
    True
    >>> cache_hash( ('foo', 55, (), (7, 'bar')) ) == cache_hash( tuple(['f' + 'oo', 60-5, tuple(), (int("7"), 'foobar'[3:], 'bazinga')[:2]]))
    True
    >>> cache_hash( list(range(3)) ) == cache_hash( (0,1,2) ) == cache_hash( [0,1,2] )
    True
    >>> cache_hash( dict(foo='bar', baz=66, boing=77) ) == cache_hash( {'boing':77, 'foo':'bar', 'baz': 66})
    True
    >>> class A: pass
    >>> a = A()
    >>> a.foo = 'bar'
    >>> b = A()
    >>> b.foo = 'b' + 'ar'
    >>> cache_hash(a) == cache_hash(b)
    True
    >>> b.x = 77
    >>> cache_hash(a) == cache_hash(b)
    False
    >>> cache_hash(True) == cache_hash(3 == 3)
    True
    >>> cache_hash(False) == cache_hash(3 == 4)
    True
    >>> class B:
    ...   def cache_hash(self): return self.x
    >>> a = B()
    >>> a.x = 10
    >>> a.y = 88
    >>> b = B()
    >>> b.x = 10
    >>> b.y = 99
    >>> cache_hash(a) == cache_hash(b)
    True
    """

    if isinstance(obj, dict):
        r = cache_hash(tuple( (k, v) for k, v in sorted(obj.items(), key=lambda p: cache_hash(p[0]))))

    # elif isinstance(obj, int):
        # r = _hash(obj)

    # elif isinstance(obj, float):
        # r = _hash(obj)

    elif obj is None:
        #r = hash(obj)
        r = 0x7f4aad69315

    elif isinstance(obj, str):
        r = cache_hash(tuple(map(ord, obj)))

    elif isinstance(obj, bytes):
        r = cache_hash(tuple(obj))

    elif isinstance(obj, list):
        r = cache_hash(tuple(obj))

    elif isinstance(obj, tuple):
        r = cache_hash(sum(map(cache_hash, obj)))

    elif np and isinstance(obj, np.ndarray):
        r = _hash(obj.data.tobytes())

    elif pd and isinstance(obj, pd.DataFrame):
        r = pd.util.hash_pandas_object(obj).sum()

    elif hasattr(obj, 'cache_hash') and callable(obj.cache_hash):
        r = obj.cache_hash()

    # Don't do this!
    # Python changes values of hash functions with every process start for security reasons,
    # these will not be useful for hashing
    # elif hasattr(obj, '__hash__') and callable(obj.__hash__):
        # r = obj.__hash__()

    elif hasattr(obj, '__class__') and hasattr(obj, '__dict__'):
        r = cache_hash( (obj.__class__.__name__, obj.__dict__) )

    elif hasattr(obj, 'to_bytes') and callable(obj.to_bytes):
        r = _hash(obj.to_bytes(100, 'big', signed=True))

    elif hasattr(obj, 'tobytes') and callable(obj.tobytes):
        r = _hash(obj.tobytes())

    else:
        raise TypeError("dont know how to hash {} of type {}".format(obj, type(obj)))
    return r

