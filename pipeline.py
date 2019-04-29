

"""
Problem statement:

Lets say we have some functions that work with large data as input and output, eg large
numpy arrays (but could by anything):

def foo(a):
    return a + 1

def bar(a):
    return a * 2

def main():
    b = foo(a)
    c = bar(b)

Now lets assume these functions are pure in the sense that the same input will always produce
the same result and at the same time they incur long running computations.
Since this is data analysis settings we would like to cache function results to disk.
Now we can put a @cached to foo() and bar(), and assuming the result is already cached then main would do this:

1. Compute the hash for a (somewhat expensive)
2. Load b from disk
3. Compute the hash of b (somewhat expensive)
4. Load c from disk

This approach has several issues:
a. Data is loaded from disk that is never used as it is input to another cached computation
b. Worse we need to calculate hash values on data that is not needed
c. Whenever a function implementation changes (which happens all the time), we might forget
   to explicitely clear the cache

c.) can possibly be addressed with some introspection (can we get a hash value of the code of a function from interpreter?)
a. + b.) Are a bit more tricky.

- Hash annotation
  We could return together with the actual data a (stored) hash value of it to pass on to the next
  cache lookup. This would still unnecessarily load the data, but avoid computation of the hashes.
  Downside is the result is a special object (value + hash), not just the value we expect.
  Maybe we can get around this with a central (ram) registry of object id -> hash. That would of course
  brake as soon as a copy is made of a returned value.

- Call graph:
  Instead of just wrapping functions into a cache handler, we could construct a call graph
  similar to eg tensorflow.
  TODO: Learn more about how tensorflow does it. Maybe they do already everything we need and
  we can just use it?
  Returned values would be parts of these graphs, explicit execution in the last step would be necessary
  to receive a value.
  This can avoid loading & hashing and give potential other benifits (like some automatic parallelization?)
  The cost are somewhat different semantics.

"""

ram_storage = {}

class RamStorage:
    def __init__(self):
        self.hashes = {}
        self.values = {}

    def __getitem__(self, key):
        return Value(hash_=self.hashes[key], value=self.values[key])

    def __setitem__(self, key, value):
        self.values[key] = value
        self.hashes[key] = hash(value)

# TODO: DiskStorage which loads value only ondemand

class Call:
    def __init__(self, op, args, kws):
        self.operation = op
        self.args = args
        self.kws = kws

    def __repr__(self):
        return f'{str(self.operation)}(args={self.args}, kws={self.kws})'

    def __hash__(self):
        return hash(self.operation)

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

    def __hash__(self):
        # TODO: proper hashing, see cache.py
        return hash((
            hash(self.call),
            # hash(tuple(self.args)),
            # hash(self.kws)
        ))

    # def load_return_value(self):

class Unknown: pass

class Value:
    def __init__(self, value=Unknown, hash_=Unknown):
        self.hash = hash_
        self.value = value

    def load_value(self):
        if self.value is not Unknown:
            return self.value

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

        try:
            result = self.storage[hash(execution)]
        except KeyError:
            result = execution.compute()
            self.storage[hash(execution)] = result

        return result


class Operation:
    def __init__(self, f):
        self.f = f
        # self.f.__code__.__hash__()

    def __call__(self, *args, **kws):
        return Call(self, args, kws)

    def __repr__(self):
        return self.f.__name__

    def __hash__(self):
        return hash(self.f.__code__)

    def compute(self, args_values, kws_values):
        return self.f(*args_values, **kws_values)


def operation():
    def wrapper(f):
        return Operation(f)
    return wrapper


def main():

    @operation()
    def foo(a):
        print("actually computing: foo a=", a)
        return a + 1

    @operation()
    def bar(a):
        print("actually computing: bar a=", a)
        return a * 2

    a = 1
    b = foo(a)
    c = bar(b)
    print(c)

    s = Session(ram_storage)
    print('compute=',s.compute(c))
    print('compute=',s.compute(c))


if __name__ == '__main__':
    main()


