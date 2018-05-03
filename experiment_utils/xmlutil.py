def element_text(ele, name, ctor = lambda x: x, default = None, ns = {}):
    e = ele.find(name, ns)
    if e is None or e.text is None:
        return default
    return ctor(e.text)

def attr_text(ele, name, ctor = lambda x: x, default = None, ns = {}):
    if name not in ele.attrib or ele.attrib[name] is None:
        return default
    return ctor(ele.attrib[name])
