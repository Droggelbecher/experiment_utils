
class DictDB:

    """
    For dictionaries of the form:

    {
        'name': "something",
        'parameter1': 44.7,

        'stats': {
            'time': [1, 1, 2, 3, ... ],
            'temperature': [88, 89, 234, 9089, ...],
            ...
        }
    }

    ds = [ { ... }, { ... }, ... ]

    db = DictDB(ds, base = lambda d: d['stats'])

    db.chain_select('time', where = { 'name': 'something' })
    db.sort_all_according_to('time')
    """

    def __init__(self, ds, base = lambda x: x):
        self.dicts = ds
        self.base = base


    def chain_select(self, keys, where = {}, order_by = None):
        #print "chain_select('{}', {})".format(key, where)
        rs = {key: [] for key in keys}

        assert order_by is None or order_by in keys

        for d in self.dicts:
            #print "d=", d.keys()
            for k, v in where.items():
                if d[k] != v:
                    break
            else:
                for key in keys:
                    rs[key].extend(self.base(d)[key])
                #r.extend(self.base(d)[key])
                #print "self.base(d).keys()={}".format(self.base(d).keys())
                #print self.base(d)[key]
                #print "key=", key
                #print "r=", r


        z = []
        if order_by is None:
            ks = list(keys)
        else:
            ks = [order_by] + list(set(keys) - set([order_by]))
        for k in ks:
            z.append(rs[k])
        z = zip(*sorted(zip(*z)))

        r = []
        for k, v in zip(keys, z):
            r.append(v)

        return r

    def sort_all_according_to(self, key):
        for d in self.dicts:

            keys = [key] + list(set(self.base(d).keys()) - set([key]))

            z = []
            for k in keys:
                z.append(self.base(d)[k])

            z = zip(*sorted(zip(*z)))
            for k, v in zip(keys, z):
                self.base(d)[k] = v


