

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
            os.path.join(
                'out/',
                (name + '-' if name else '') + now
            )
        )

    def create(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def dump(self, filename, data):
        self.create()
        with open(os.path.join(self.path, filename + '.json'), 'w') as f:
            json.dump(data, f, cls = DecimalEncoder)

