import os
from experiment_utils.hashing import cache_hash
import pickle

class Unknown:
    @classmethod
    def __bool__(cls, self):
        return False

class Value:
    def __init__(self, value=Unknown, hash_=Unknown):
        self._hash = hash_
        self._value = value

    def load_value(self):
        assert self._value is not Unknown
        return self._value

    def get_hash(self):
        if self._hash is Unknown:
            self._hash = cache_hash(self._value)
        return self._hash

    def cache_hash(self):
        return self.get_hash()

class RamStorage:
    def __init__(self):
        self.hashes = {}
        self.values = {}

    def load(self, key):
        if key not in self.hashes:
            raise KeyError()
        return Value(hash_=self.hashes[key], value=self.values[key])

    def save(self, key, value):
        if isinstance(value, Value):
            self.values[key] = value.value
            self.hashes[key] = value.hash
        else:
            self.values[key] = value
            self.hashes[key] = cache_hash(value)

class DiskStorage:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def _get_hash_filename(self, key):
        return self.base_dir + '/' + hex(abs(key)) + '_hash.p'

    def _get_value_filename(self, key):
        return self.base_dir + '/' + hex(abs(key)) + '_value.p'

    def load(self, key):
        filename = self._get_hash_filename(key)
        if not os.path.exists(filename):
            raise KeyError()

        with open(filename, 'rb') as f:
            h = pickle.load(f)

        def load_value():
            filename = self._get_value_filename(key)
            with open(filename, 'rb') as f:
                v = pickle.load(f)
            return v

        r = Value(hash_=h)
        r.load_value = load_value
        return r


    def save(self, key, value):
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        with open(self._get_hash_filename(key), 'wb') as f:
            pickle.dump(value.get_hash(), f)

        with open(self._get_value_filename(key), 'wb') as f:
            pickle.dump(value.load_value(), f)

class Call:
    def __init__(self, op, args, kws):
        self.operation = op
        self.args = args
        self.kws = kws

    def __repr__(self):
        return f'{str(self.operation)}(args={self.args}, kws={self.kws})'

    def cache_hash(self):
        return cache_hash(self.operation)

class CallExecution:
    def __init__(self, call, args, kws):
        self.call = call
        self.args = args
        self.kws = kws

    def compute(self):
        args_values = [a.load_value() for a in self.args]
        kws_values = {k: v.load_value() for k, v in self.kws.items()}

        return Value(value=self.call.operation.compute(args_values, kws_values))

    def __repr__(self):
        return repr(self.call) + repr(self.args) + repr(self.kws)

    def cache_hash(self):
        return cache_hash((
            cache_hash(self.call),
            cache_hash(self.args),
            cache_hash(self.kws)
        ))

class Session:
    def __init__(self, storage):
        self.storage = storage

    def compute(self, call):
        if not isinstance(call, Call):
            # Not a call but a constant, just return it wrapped in a value
            return Value(value=call)

        # args/kws contain CallExecutions
        args = [self.compute(a) for a in call.args]
        kws = {k: self.compute(v) for k, v in call.kws.items()}

        execution = CallExecution(call, args=args, kws=kws)
        key = cache_hash(execution)

        try:
            result = self.storage.load(key)
        except KeyError:
            result = execution.compute()
            self.storage.save(key, result)

        return result


class Operation:
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kws):
        return Call(self, args, kws)

    def __repr__(self):
        return self.f.__name__

    def cache_hash(self):
        r = cache_hash(self.f.__code__.co_code)
        return r

    def compute(self, args_values, kws_values):
        return self.f(*args_values, **kws_values)


def operation():
    def wrapper(f):
        return Operation(f)
    return wrapper


def main():

    import numpy as np

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


if __name__ == '__main__':
    main()


