
from collections import OrderedDict
import logging

try:
    import ijson.backends.yajl2_cffi as ijson
    logging.info("using yajl2/cffi parser :-)")
except ImportError:
    try:
        import ijson.backends.yajl2 as ijson
        logging.warn("using yajl2/ctypes parser (you can do better by installing python-cffi!)")
    except ImportError:
        try:
            import ijson.backends.yajl as ijson
            logging.warn("using yajl/ctypes parser (try installing yajl2 and ideally cffi!)")
        except ImportError:
            import ijson
            logging.warn("warning: using python ijson implementation. This will be slow!")

class InterruptParsing(Exception):
    def __init__(self, popped_parser):
        self.popped_parser = popped_parser

class IjsonParser:
    def __init__(self, lexer, parent = None):
        self.structure = []
        self.keys = []
        self.lexer = lexer
        self.child = None
        self.parent = parent

    def on_start_map(self, value):
        self.structure.append(OrderedDict())
        yield from ()

    def on_map_key(self, value):
        self.keys.append(value)
        yield from ()

    def on_end_map(self, value):
        yield from self.on_value(self.structure.pop())

    def on_start_array(self, value):
        self.structure.append([])
        yield from ()

    def on_end_array(self, value):
        yield from self.on_value(self.structure.pop())

    def on_string(self, value):
        yield from self.on_value(value)

    def on_number(self, value):
        yield from self.on_value(value)

    def on_boolean(self, value):
        yield from self.on_value(value)

    def on_null(self, value):
        yield from self.on_value(value)

    def receive_from_child(self, value):
        yield from self.emit(value)

    def top_parser(self):
        p = self
        while p.child:
            p = p.child
        return p

    def push_parser(self, parser):
        p = self.top_parser()
        assert parser is not p
        p.child = parser
        parser.parent = p

    def pop_parser(self):
        p = self.top_parser()
        if p.parent:
            p.parent.child = None
            p.parent = None
            raise InterruptParsing(p)

    def emit(self, value):
        """
        Send value to parent parser, or yield it if self is the topmost parser
        :param value: Value to send upwards
        """
        if self.parent:
            yield from self.parent.receive_from_child(value)
        else:
            yield value, self
        yield from ()

    def on_value(self, value):
        if not len(self.structure):
            yield from self.emit(value)
            self.pop_parser()
        else:
            last = self.structure[-1]
            if isinstance(last, OrderedDict):
                last[self.keys.pop()] = value
            elif isinstance(last, list):
                last.append(value)

    def send(self, event, value):
        yield from getattr(self.top_parser(), 'on_' + event)(value)

    def parse_all(self, allow_skip = False):
        if self.top_parser() is not self:
            assert False, (self.top_parser(), self)
        for event, value in self.lexer:
            try:
                for r, source in self.send(event, value):
                    if source is self:
                        yield r
                    else:
                        if not allow_skip:
                            raise UserWarning('ignoring {} from {} (self={})'.format(r, source, self))
                        else:
                            print("ignoring {} from {} (self={})".format(r, source, self))
            except InterruptParsing as e:
                if e.popped_parser is self:
                    break

def test_ijson_parser():
    from io import BytesIO

    sio = BytesIO(bytearray("""
    {
        "foo": "bar",
        "foo2": [1, 2, 3],
        "bar": false,
        "something": null,
        "dct": { "foo": [[[], [[]]]], "bar": {"a": {}, "b": { "c": {}}} }
    }
    """, 'utf-8'))

    p = IjsonParser(ijson.basic_parse(sio))
    for v in p.parse_all():
        print(v)

    sio = BytesIO(bytearray("""
    [
    { "id": 1 },
    { "id": 2 },
    { "id": 3 }
    ]
    """, 'utf-8'))
    p = IjsonParser(ijson.basic_parse(sio))
    for v in p.parse_all():
        print(v)

if __name__ == '__main__':
    test_ijson_parser()


