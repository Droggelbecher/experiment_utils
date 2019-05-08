
import os
import pickle
from icecream import ic

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

    def _get_meta_filename(self, key):
        return self.base_dir + '/' + hex(abs(key)) + '_meta.p'

    def load(self, key):
        from .calltree import Value

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

    @staticmethod
    def load_meta(filename):
        if not os.path.exists(filename):
            raise KeyError()
        with open(filename, 'rb') as f:
            h = pickle.load(f)
        return h


    def save(self, key, value, meta={}):
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        with open(self._get_hash_filename(key), 'wb') as f:
            pickle.dump(value.get_hash(), f)

        with open(self._get_value_filename(key), 'wb') as f:
            pickle.dump(value.load_value(), f)

        meta = dict(meta)
        meta['hash_filename'] = self._get_hash_filename(key)
        meta['value_filename'] = self._get_value_filename(key)
        meta['meta_filename'] = self._get_meta_filename(key)

        with open(self._get_meta_filename(key), 'wb') as f:
            pickle.dump(meta, f)

