#!/usr/bin/env python3

import sys
import os
import os.path
import pickle
import hashlib
import time
import shutil
import logging

WRITE_CACHE = True
READ_CACHE = True

base_directory = os.path.abspath('_cache')

def add_arguments(parser):

    parser.add_argument(
        '--no-cache',
        help = 'disable reading and writing of cached values'
    )

    return parser

def process_arguments(args):

    if args.no_cache:
        WRITE_CACHE = False
        READ_CACHE = False


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

    elif isinstance(obj, int):
        r = hash(obj)

    elif isinstance(obj, float):
        r = hash(obj)

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
        r = hash(tuple(map(cache_hash, obj)))

    elif hasattr(obj, 'cache_hash') and callable(obj.cache_hash):
        r = obj.cache_hash()

    elif hasattr(obj, '__class__') and hasattr(obj, '__dict__'):
        r = cache_hash( (obj.__class__.__name__, obj.__dict__) )
    else:
        raise TypeError("dont know how to hash {} of type {}".format(obj, type(obj)))

    logging.debug("hashing {} of type {} --> {}".format(obj, type(obj), r))
    return r



def _cache_name(f, args, kws, ignore_kws=()):
    items = tuple(x for x in sorted(kws.items(), key=lambda p: cache_hash(p[0])) if x[0] not in ignore_kws)
    logging.debug("hashing items: {}".format([i[0] for i in items]))
    return f.__name__ + '-{:08x}'.format( cache_hash( (args, items) ))

def _cache_path(f, args, kws, ignore_kws):
    return os.path.join(base_directory, _cache_name(f, args, kws, ignore_kws))

CACHE_AVAILABLE, CACHE_NOT_AVAILABLE, CACHE_COLLISION = tuple(range(3))

def _verify_cache(cache_data, f, kws, ignore_kws):
    items = tuple(x for x in sorted(kws.items(), key=lambda p: cache_hash(p[0])) if x[0] not in ignore_kws)
    items2 = tuple(x for x in sorted(cache_data['kws'].items(), key=lambda p: cache_hash(p[0])) if x[0] not in ignore_kws)
    return (
            cache_data['function_name'] == f.__name__ and
            items == items2
    )

def _get_from_cache(cache_path, f, kws, ignore_kws, filenames):
    if not READ_CACHE:
        return CACHE_NOT_AVAILABLE

    if os.path.exists(cache_path):
        # A matching cache file exists!
        # Now find out whether it is up-to-date
        cache_file = open(cache_path, 'rb')
        logging.debug("cache file: {}".format(cache_path))
        cache_data = pickle.load(cache_file)

        if not _verify_cache(cache_data, f, kws, ignore_kws):
            logging.warning("cache hash collision for {}({}) (wrongly maps to {}), not loading from there!".format(f.__name__, kws, cache_path))
            return CACHE_COLLISION, None

        for filename in filenames:
            try:
                if os.stat(filename).st_mtime > cache_data['timestamp']:
                    logging.debug("answer for {}({}) in {} outdated, recomputing".format(f.__name__, kws, cache_path))
                    break
            except FileNotFoundError:
                # Hmm file is not there.
                # Dunno if it was there before. Lets default to recomputing
                # (probably fast when an input file is missing)
                logging.warning("input file {} not found for timestamp check, recomputing".format(filename))
                break
        else:
            cache_file.close()
            return CACHE_AVAILABLE, cache_data
        cache_file.close()
    return CACHE_NOT_AVAILABLE, None

def _clear_cache(cache_path):
    os.remove(cache_path)

def clear_caches():
    try:
        shutil.rmtree(base_directory)
    except FileNotFoundError:
        pass

ALWAYS = lambda kws: True
NEVER = lambda kws: False

def cached(filename_kws=(), ignore_kws=(), add_filenames=(), cache_if=ALWAYS, compute_if=ALWAYS, cache_exception = NEVER):
    """
    >>> @cached()
    ... def f(x):
    ...   print("calculating f(" + str(x) + ")")
    ...   return x * x

    >>> clear_caches()

    >>> f(x = 5)
    calculating f(5)
    25
    >>> f(x = 2 + 3)
    25
    >>> f(x = 2)
    calculating f(2)
    4
    >>> f(x = 5)
    25
    >>> f(x = 2)
    4
    """

    def decorate(f):
        def new_f(**kws):
            add_filenames_ = add_filenames
            cache_path = _cache_path(f, (), kws, ignore_kws)

            # Which files determine whether our result is up to date?

            filenames = set(kws[k] for k in filename_kws)
            if callable(add_filenames_):
                add_filenames_ = add_filenames_(kws)
            filenames = filenames.union(set(add_filenames_))

            # Get from cache if present and fresh

            cache_result, cache_data = _get_from_cache(cache_path, f, kws, ignore_kws, filenames)
            if cache_result == CACHE_AVAILABLE:
                logging.debug("answering {}({}) from {}".format(f.__name__, kws, cache_path))
                e = cache_data.get('exception', None)
                if e is not None:
                    raise e
                return cache_data['return_value']

            # Compute

            exception = None
            if compute_if(kws):
                t = time.time()

                if cache_exception(kws):
                    try:
                        r = f(**kws)
                    except Exception as e:
                        exception = e
                else:
                    r = f(**kws)

                dt = time.time() - t

            else:
                logging.error("Cannot answer {}({}) from path {} and compute_if() returned False".format(f.__name__, kws, cache_path))
                raise Exception()

            # Save to cache

            # Exception case
            if exception is not None and cache_result != CACHE_COLLISION and (WRITE_CACHE and cache_if(kws)):
                reduced_kws = dict(kws)
                for k in ignore_kws:
                    del reduced_kws[k]

                cache_data = {
                    'timestamp':  time.time(),
                    'computation_time': dt,
                    'function_name':  f.__name__,
                    'kws': reduced_kws,
                    'return_value':  None,
                    'exception': e
                }
                if not os.path.exists(base_directory):
                    os.mkdir(base_directory)

                logging.debug("caching {}({}) [{:.2f}s] -> {}".format(f.__name__, kws, dt, cache_path))
                cache_file = open(cache_path, 'wb')
                pickle.dump(cache_data, cache_file)
                cache_file.close()
                raise e

            # "Normal" case
            elif cache_result != CACHE_COLLISION and (WRITE_CACHE and cache_if(kws)):
                reduced_kws = dict(kws)
                for k in ignore_kws:
                    del reduced_kws[k]

                cache_data = {
                    'timestamp':  time.time(),
                    'computation_time': dt,
                    'function_name':  f.__name__,
                    'kws': reduced_kws,
                    'return_value':  r,
                    'exception': None,
                }

                if not os.path.exists(base_directory):
                    os.mkdir(base_directory)

                logging.debug("caching {}({}) [{:.2f}s] -> {}".format(f.__name__, kws, dt, cache_path))
                cache_file = open(cache_path, 'wb')
                pickle.dump(cache_data, cache_file)
                cache_file.close()
            return r
        return new_f
    return decorate


