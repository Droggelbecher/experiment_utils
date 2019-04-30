
import numpy as np
from collections.abc import Sequence

class DataFrame:
    """
    - All columns and only columns have (string) names
    """

    def __init__(self, a=(), columns=(), *, data={}, copy=True, dtype=None):
        self._columns = {}

        if columns:
            a = np.array(a).reshape((-1, len(columns)))
            for i, name in enumerate(columns):
                self._columns[name] = np.array(a[:, i], copy=copy, dtype=dtype)

        self._columns.update({
            name: np.array(v, copy=copy, dtype=dtype) for name, v in data.items()
        })
        self.names = tuple(self._columns.keys())
        self._index = None

    def copy(self):
        return DataFrame(data=self._columns, copy=True)

    @property
    def dtype(self):
        return [(k, v.dtype) for (name, v) in self._columns]

    def __len__(self):
        try:
            # print('__len__', len(tuple(self._columns.values())[0]))
            return len(tuple(self._columns.values())[0])
        except IndexError:
            return 0

    def _normalize_index(self, index):
        def is_seq(x):
            return isinstance(x, Sequence) or isinstance(x, np.ndarray)

        if type(index) is not tuple:
            if (type(index) is str) or (is_seq(index) and type(index[0]) is str):
                return self._normalize_index((slice(None), index))
            return self._normalize_index((index, slice(None)))

        row, col = index

        # if not isinstance(row, slice) and not is_seq(row):
        if isinstance(row, int):
            row = slice(row, row+1)

        if isinstance(col, slice):
            assert col.start is None and col.stop is None
            col = tuple(self.names)

        elif isinstance(col, str):
            col = (col,)

        if isinstance(row, Sequence) and len(row) > len(self):
            raise IndexError

        if isinstance(row, slice) and (
                    (row.start is not None and (row.start < 0))
                 or (row.stop is not None and (row.stop > len(self)))
        ):
            raise IndexError

        # At this point row is a slice or sequence
        # and col is a sequence of column names
        return (row, col)

    def __getitem__(self, index):
        row, col = self._normalize_index(index)

        # if len(col) == 1:
            # return self._columns[col[0]][row]

        return DataFrame(
            data = {k: self._columns[k][row] for k in col},
            copy = False
        )

    def __setitem__(self, index, value):
        row, col = self._normalize_index(index)

        full_column = isinstance(row, slice) and row.start is None and row.stop is None

        if isinstance(value, np.ndarray):
            for i, k in enumerate(col):
                if full_column:
                    self._columns[k] = value[i]
                else:
                    self._columns[k][row] = value[i]

        else:
            for k in col:
                self._columns[k][row] = value

        if self._index is not None and self._index in col:
            self._sustain_index()

    def filter_or(self, filters=[]):
        """
        >>> df = DataFrame([[1,2], [3,4], [6,5]], columns=['a', 'b'])
        >>> df.or([('a', lambda a: a % 2 != 0), ('b', lambda b: b >= 3)], 'and')
        """
        n = len(self)
        mask = np.full(n, False)
        for c, f in filters:
            mask[~mask] = f(self._columns[c][~mask])
        return self[mask, :]

    def filter_and(self, filters=[]):
        """
        >>> df = DataFrame([[1,2], [3,4], [6,5]], columns=['a', 'b'])
        >>> df.and([('a', lambda a: a % 2 != 0), ('b', lambda b: b >= 3)], 'and')
        """
        n = len(self)
        mask = np.full(n, True)
        for c, f in filters:
            mask[mask] = f(self._columns[c][mask])
        return self[mask, :]

    def array(self, columns=None):
        if columns is None:
            columns = self.names

        return np.vstack([self._columns[k] for k in columns])

    def _sustain_index(self, copy=True):
        idx = np.argsort(self._columns[self._index])
        for k in self.names:
            self._columns[k] = np.array(self._columns[k][idx], copy=copy)


    def make_index(self, col, is_sorted=False, copy=True):
        self._index = col
        if not is_sorted:
            self._sustain_index(copy=copy)

    def get_range(self, start, stop=None):
        assert self._index is not None

        if stop is None:
            stop = start
        c = self._columns[self._index]
        r = slice(
            np.searchsorted(c, start, 'left'),
            np.searchsorted(c, stop, 'right'),
        )
        return self[r, :]

    def _to_array(self, other):
        if isinstance(other, DataFrame):
            return other.array()
        return other

    def __add__(self, other):
        return self.array() + self._to_array(other)

    def __sub__(self, other):
        return self.array() - self._to_array(other)

    def __mul__(self, other):
        return self.array() * self._to_array(other)

    def __div__(self, other):
        return self.array() / self._to_array(other)

    def __str__(self):
        from .text import format_table
        columns = self.names
        return format_table(
            [
                [((i < len(self._columns[k])) and self._columns[k][i] or '-') for k in columns]
                for i in range(len(self))
            ],
            headers = columns
        )


if __name__ == '__main__':

    df = DataFrame(
        np.array([[7,2,3], [4,5,6], [5,8,9]]),
        ('foo','bar','baz')
    )
    # print(df)
    # print(df[1])
    # print(df['bar'])
    # print(df[1, :])
    # print(df[:, 'bar'])

    print('Whole array')
    print(df[['baz', 'bar', 'foo']])
    print(df[:, :])
    print(df.array())

    print('Indexing')
    df[:, 'bar'] = 777
    df[1, :] = 555
    df[1, 'bar'] = 666
    print(df)
    print(df[[True, False, True], :])

    print('Filtering')
    print(df.filter_and([
        ('baz', lambda x: x >= 7),
        ('bar', lambda x: x <= 700),
    ]))

    print('Making an index for fast access')

    df = DataFrame(
        np.random.rand(50, 2),
        ('t', 'value')
    )
    df['t'] = np.floor(df['t'].array() * 10.0)
    print(df)
    df.make_index('t')
    # This sorts by that column...
    print(df)
    print(df.get_range(5.0))
    print(df.get_range(5.0, 7.0))

    # Will automagically make sure this stays sorted by 't']
    df[10:40] = np.array([12.0, 999.9]).T
    print(df)

    print("Reference/Copy semantics")
    df = DataFrame(
        np.array([[7,2,3], [4,5,6], [5,8,9]]),
        ('foo','bar','baz')
    )
    print(df)
    s = df[1, :]
    s['foo'] = 999 # This actually modifies df
    print(df)
    s = df[2, :].copy()
    s['foo'] = 333 # This doesn't
    print(df)


    # df[1, 'a']
    # df[:, 'a']

