
import numpy as np

def sparsly(a):
    s = 'ar[\n  shape: {}\n'.format(a.shape)

    idxs = np.where(a != 0)
    for t in zip(*idxs):
        s += '  {} = {}\n'.format(t, a[t])
    return s + ']'

def mesh_cartesian(mesh, s):
    """
    #>>> import numpy as np
    #>>> mesh1 = np.meshgrid( [1,2,3], [10, 100] )
    #>>> mesh2 = np.meshgrid( [1,2,3], [10, 100], [5,6,7] )
    #>>> mesh11 = extend_mesh(mesh1, [5,6,7])
    #>>> mesh11 == mesh2
    #True

    >>> m0 = np.meshgrid( [1,2], [4,5,6] )
    >>> r = mesh_cartesian(m0, [ [0,0,1], [0,1,0], [1,0,0] ])
    >>> np.array(r).T.reshape(-1, 5)
    array([[1, 4, 0, 0, 1],
           [1, 5, 0, 1, 0],
           [1, 6, 1, 0, 0],
           [2, 4, 0, 0, 1],
           [2, 5, 0, 1, 0],
           [2, 6, 1, 0, 0],
           [1, 4, 0, 0, 1],
           [1, 5, 0, 1, 0],
           [1, 6, 1, 0, 0],
           [2, 4, 0, 0, 1],
           [2, 5, 0, 1, 0],
           [2, 6, 1, 0, 0],
           [1, 4, 0, 0, 1],
           [1, 5, 0, 1, 0],
           [1, 6, 1, 0, 0],
           [2, 4, 0, 0, 1],
           [2, 5, 0, 1, 0],
           [2, 6, 1, 0, 0]])
    """
    s = np.array(s)

    # To each array in mesh, add one new dimension (as last),
    # that repeats the value len(s) times
    r = [
        np.repeat(m, s.shape[0], len(m.shape) - 1)
        .reshape(m.shape + (s.shape[0],))
        for m in mesh
    ]

    print ('m0 sh', mesh[0].shape)
    print ('s sh', s.shape)
    print (np.repeat(
            s.reshape(1, -1),
            np.prod(mesh[0].shape),
            0
        ))

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
                    ss,
                    np.prod(mesh[0].shape),
                    0
                )
                .reshape(mesh[0].shape + (ss.shape[0],))
            )

    return r


def mesh_cartesian_seq(mesh, s):
    s = np.array(s)

    # To each array in mesh, add one new dimension (as last),
    # that repeats the value len(s) times
    r = [
        np.repeat(m, s.shape[0], len(m.shape) - 1)
        .reshape(m.shape + (s.shape[0],))
        for m in mesh
    ]

    print ('m0 sh', mesh[0].shape)
    print ('s sh', s.shape)
    print (np.repeat(
            s.reshape(1, -1),
            np.prod(mesh[0].shape),
            0
        ))

    r.append(
        np.repeat(
            s.reshape(1, -1),
            np.prod(mesh[0].shape),
            0
        )
        .reshape(mesh[0].shape + (s.shape[0],))
    )

    return r



def all_onehot(n):
    return np.diag(np.ones(n))

