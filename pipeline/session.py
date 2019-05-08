
from hashing import cache_hash

class Session:
    def __init__(self, storage):
        self.storage = storage

    def compute(self, call):
        from .calltree import Call, Value, CallExecution

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
            self.storage.save(key, result, meta=execution.get_metadata(args, kws))

        return result


