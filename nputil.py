
import numpy as np

def moving_average(a, n=3) :
    """
    Moving average of width n (entries) on a numpy array.
    source: http://stackoverflow.com/questions/14313510/moving-average-function-on-numpy-scipy
    """
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = (ret[n:] - ret[:-n]) / n
    ret[:n] = ret[:n] / (np.arange(n) + 1)
    return ret


def shuffle_rows(*args):
    """
    Given a number of lists/tuples/NP-arrays, return consistently row-shuffled
    versions of these (all shuffled with the same row-index permutation).

    >>> A = np.array([
    ...    [0, 1, 2],
    ...    [3, 4, 5],
    ...    [6, 7, 8],
    ...    [9, 10, 11]
    ... ])
    >>> B = A * 10
    >>> l = [ "one", "two", "three", "four" ]
    >>> i = np.arange(4)
    >>> A2, B2, l2, i2 = shuffle_rows(A, B, l, i)

    Return values are NP-array versions of the input iterables
    that are all shuffled in the same way, i2 in this case
    returns exactly that index permutation.

    >>> len(i2) == 4
    True
    >>> np.all(np.sort(i2) == i)
    True
    >>> np.all(A[i2] == A2)
    True
    >>> np.all(B[i2] == B2)
    True
    >>> np.all(np.array(l)[i2] == np.array(l2))
    True
    """

    a = np.arange(args[0].shape[0])
    np.random.shuffle(a)
    return [np.array(arg)[a] for arg in args]

def sparsly(a):
    """
    Return a string representation of a (for printing),
    that only lists non-zero fields.
    """
    s = 'ar[\n  shape: {}\n'.format(a.shape)
    idxs = np.where(a != 0)
    for t in zip(*idxs):
        s += '  {} = {}\n'.format(t, a[t])
    return s + ']'

def mesh_cartesian(mesh, s):
    """
    Extend the given mesh grid with the given sequence

    >>> m0 = np.meshgrid( [1,2], [4,5,6] )
    >>> m0
    [array([[1, 2],
           [1, 2],
           [1, 2]]), array([[4, 4],
           [5, 5],
           [6, 6]])]
    >>> np.array(m0).T.reshape(-1, 2)
    array([[1, 4],
           [1, 5],
           [1, 6],
           [2, 4],
           [2, 5],
           [2, 6]])

    One-dimensional case:

    >>> r = mesh_cartesian(m0, [11, 22, 33])
    >>> r_expected = [
    ...   np.array([
    ...     [[1, 1, 1], [2, 2, 2]],
    ...     [[1, 1, 1], [2, 2, 2]],
    ...     [[1, 1, 1], [2, 2, 2]]
    ...   ]),
    ...   np.array([
    ...     [[4, 4, 4], [4, 4, 4]],
    ...     [[5, 5, 5], [5, 5, 5]],
    ...     [[6, 6, 6], [6, 6, 6]]
    ...   ]),
    ...   np.array([
    ...     [[11, 22, 33], [11, 22, 33]],
    ...     [[11, 22, 33], [11, 22, 33]],
    ...     [[11, 22, 33], [11, 22, 33]]
    ...   ])
    ... ]

    >>> np.all(np.array(r) == np.array(r_expected))
    True

    >>> np.array(r).T.reshape(-1, 3)
    array([[ 1,  4, 11],
           [ 1,  5, 11],
           [ 1,  6, 11],
           [ 2,  4, 11],
           [ 2,  5, 11],
           [ 2,  6, 11],
           [ 1,  4, 22],
           [ 1,  5, 22],
           [ 1,  6, 22],
           [ 2,  4, 22],
           [ 2,  5, 22],
           [ 2,  6, 22],
           [ 1,  4, 33],
           [ 1,  5, 33],
           [ 1,  6, 33],
           [ 2,  4, 33],
           [ 2,  5, 33],
           [ 2,  6, 33]])


    Now with a sequence of rows as second parameter.
    Result must look like this:

    >>> r_expected = [
    ...   np.array([
    ...     [[1, 1, 1], [2, 2, 2]],
    ...     [[1, 1, 1], [2, 2, 2]],
    ...     [[1, 1, 1], [2, 2, 2]]
    ...   ]),
    ...   np.array([
    ...     [[4, 4, 4], [4, 4, 4]],
    ...     [[5, 5, 5], [5, 5, 5]],
    ...     [[6, 6, 6], [6, 6, 6]]
    ...   ]),
    ...   np.array([
    ...     [[0, 0, 33], [0, 0, 33]],
    ...     [[0, 0, 33], [0, 0, 33]],
    ...     [[0, 0, 33], [0, 0, 33]]
    ...   ]),
    ...   np.array([
    ...     [[0, 22, 0], [0, 22, 0]],
    ...     [[0, 22, 0], [0, 22, 0]],
    ...     [[0, 22, 0], [0, 22, 0]]
    ...   ]),
    ...   np.array([
    ...     [[11, 0, 0], [11, 0, 0]],
    ...     [[11, 0, 0], [11, 0, 0]],
    ...     [[11, 0, 0], [11, 0, 0]]
    ...   ]),
    ... ]

    >>> np.array(r_expected).T.reshape(-1, 5)
    array([[ 1,  4,  0,  0, 11],
           [ 1,  5,  0,  0, 11],
           [ 1,  6,  0,  0, 11],
           [ 2,  4,  0,  0, 11],
           [ 2,  5,  0,  0, 11],
           [ 2,  6,  0,  0, 11],
           [ 1,  4,  0, 22,  0],
           [ 1,  5,  0, 22,  0],
           [ 1,  6,  0, 22,  0],
           [ 2,  4,  0, 22,  0],
           [ 2,  5,  0, 22,  0],
           [ 2,  6,  0, 22,  0],
           [ 1,  4, 33,  0,  0],
           [ 1,  5, 33,  0,  0],
           [ 1,  6, 33,  0,  0],
           [ 2,  4, 33,  0,  0],
           [ 2,  5, 33,  0,  0],
           [ 2,  6, 33,  0,  0]])

    >>> r = mesh_cartesian(m0, [ [0,0,11], [0,22,0], [33,0,0] ])
    >>> np.all( np.array(r) == np.array(r_expected) )
    True
    """
    s = np.array(s)

    # To each array in mesh, add one new dimension (as last),
    # that repeats the value len(s) times
    r = [
        np.repeat(m, s.shape[0], len(m.shape) - 1)
        .reshape(m.shape + (s.shape[0],))
        for m in mesh
    ]

    if len(s.shape) == 1:
        r.append(
            np.repeat(
                s.reshape(1, -1),
                np.prod(mesh[0].shape),
                0
            )
            .reshape(mesh[0].shape + (s.shape[0],))
        )

    else:
        for ss in s.T:
            r.append(
                np.repeat(
                    ss.reshape(1, -1),
                    np.prod(mesh[0].shape),
                    0
                )
                .reshape(mesh[0].shape + (ss.shape[0],))
            )

    return r



def all_onehot(n):
    """
    Return a numpy array of all n possible one-hot configurations
    with n fields. (That is, the nxn matrix with all 1s on the diagonal
    """
    return np.diag(np.ones(n))

def coordinates_to_offsets(a):
    """
    Given an array a that represents a sequence of positions,
    return its delta-encoding, that is its origin (a[0]) and a sequence of
    offsets (each relative to the previous offset or the origin).

    Return: Pair (origin, offsets)

    >>> a = np.array([(1, 2), (5, 8), (10, 10)])
    >>> origin, offsets = coordinates_to_offsets(a)
    >>> origin
    array([1, 2])
    >>> offsets
    array([[4, 6],
           [5, 2]])
    """
    origin = a[0]
    offsets = a[1:] - a[:-1]
    return origin, offsets

def offsets_to_coordinates(origin, offsets):
    """
    Inverse operation of coordinates_to_offsets()

    >>> origin = np.array([1, 2])
    >>> offsets = np.array([(4, 6), (5, 2)])
    >>> a = offsets_to_coordinates(origin, offsets)
    >>> a
    array([[ 1,  2],
           [ 5,  8],
           [10, 10]])
    """

    return np.concatenate((
        np.array([origin]),
        origin + np.cumsum(offsets, axis = 0)
    ))

def to_regular(a, dtype=None):
    """
    Convert a structured array $a to a regular ndarray (view).
    """
    if dtype is None:
        dtype = a.dtype[0]

    fields = a.dtype.names
    shape = a.shape + (len(fields),)
    offsets = [a.dtype.fields[name][1] for name in fields]
    assert not np.any(np.diff(offsets, n=2))
    strides = a.strides + (offsets[1] - offsets[0],)
    r = np.ndarray(shape=shape, dtype=dtype, buffer=a, offset=offsets[0], strides=strides)

    assert r.shape == (len(a), len(a.dtype.names)), (r.shape, len(a), len(a.dtype.names))
    return r

def to_regular_copy(a, dtype=None):
    """
    Convert a structured array $a to a regular ndarray as a copy.
    """
    if dtype is None:
        dtype = a.dtype[0]
    assert np.all([dtype == a.dtype[i] for i in range(len(a.dtype))]), a.dtype
    l = [a[n] for n in a.dtype.names]
    r = np.vstack(l).T
    assert r.shape == (len(a), len(a.dtype.names)), (r.shape, len(a), len(a.dtype.names))
    return r

def to_structured(a, dtype):
    """
    Convert an ndarray $a to a structured array with the given $dtype.
    """
    return np.core.records.fromarrays(a.T, dtype=dtype)


def and_(a, *lambdas):
    """
    Logical AND for numpy index computations that uses short-circuit evaluation
    to evaluate later conditions on shorter arrays.

    >>> gt5 = lambda a: a > 5
    >>> even_count = 0
    >>> def even(a):
    ...   global even_count
    ...   even_count += len(a)
    ...   return a % 2 == 0
    >>> a = np.arange(10)
    >>> np.all(and_(a, gt5, even) == np.array([False, False, False, False, False, False, True, False, True, False]))
    True
    >>> even_count
    4
    >>> even_count = 0
    >>> np.all((gt5(a) & even(a)) == np.array([False, False, False, False, False, False, True, False, True, False]))
    True
    >>> even_count
    10
    """
    r = np.full(len(a), True)
    for l in lambdas:
        r[r] = l(a[r])
    return r

def or_(a, *lambdas):
    """
    Logical OR for numpy index computations that uses short-circuit evaluation
    to evaluate later conditions on shorter arrays.

    >>> gt5 = lambda a: a > 5
    >>> even_count = 0
    >>> def even(a):
    ...   global even_count
    ...   even_count += len(a)
    ...   return a % 2 == 0
    >>> a = np.arange(10)
    >>> np.all(or_(a, gt5, even) == np.array([True, False, True, False, True, False, True, True, True, True]))
    True
    >>> even_count
    6
    >>> even_count = 0
    >>> np.all((gt5(a) | even(a)) == np.array([True, False, True, False, True, False, True, True, True, True]))
    True
    >>> even_count
    10
    """
    r = np.full(len(a), False)
    for l in lambdas:
        r[~r] = l(a[~r])
    return r
