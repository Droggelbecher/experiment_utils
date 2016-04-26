
from collections import defaultdict

class C:
    def __init__(self, **kws):
        self.__dict__.update(kws)

class listdict(defaultdict):

    def __init__(self):
        defaultdict.__init__(self, list)

    def update(self, other):
        for k, v in other.items():
            self[k].extend(v)


