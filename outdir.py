

import os
from datetime import datetime
import json
import decimal

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        return super(DecimalEncoder, self).default(o)

class Outdir:

    def __init__(self, name = ''):
        now = datetime.now().strftime('%Y-%m-%d--%H-%M')
        self.path = os.path.abspath(
            'out-' + (name + '-' if name else '') + now
            )
        os.makedirs(self.path)

    def dump(self, filename, data):
        with open(os.path.join(self.path, filename + '.json'), 'w') as f:
            json.dump(data, f, cls = DecimalEncoder)

