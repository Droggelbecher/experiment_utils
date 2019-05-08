
from hashing import cache_hash

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

    def get_metadata(self, args, kws):
        return {
            'function': self.call.operation.f.__qualname__,
            'args': [repr(a.load_value()) for a in args],
            'kws': {k: repr(v.load_value()) for k, v in kws.items()},
        }

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



