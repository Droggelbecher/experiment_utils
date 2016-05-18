#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import numpy as np

def generate_html_bar_graph(heights, names = None):
    w = 30
    height = 100

    width = len(heights) * (w + 4)

    min_ = min(heights)
    max_ = max(heights)

    hs = [height*h/max_ if h > 0 else -height*h/min_ for h in heights]

    r = ''

    r += '<div style="width: {}px; height: {}px; border: 1px solid #a0a0a0">'.format(width, height)
    for h in hs:
        r += '<div style="height: {}px; width: {}px; margin: 2px; display: inline-block; position: relative; background-color: #00ff00; vertical-align: baseline;"></div>'.format(h if h > 0 else 0, w)
    r += '</div>'

    if min_ < 0:
        r += '<div style="width: {}px; height: {}px; border: 1px solid #a0a0a0">'.format(width, height)
        for h in hs:
            r += '<div style="height: {}px; width: {}px; margin: 2px; display: inline-block; position: relative; background-color: #ff0000; vertical-align: top;"></div>'.format(-h if h < 0 else 0, w)
        r += '</div>'


    if names is not None:
        r += '<div style="width: {}px; border: 1px solid #a0a0a0">'.format(width)
        for name in names:
            r += '<div style="width: {}px; margin: 2px; display: inline-block; position: relative; vertical-align: baseline;">{}</div>'.format(w, name)
        r += '</div>'

    r += '<br />Min: {}<br />Max: {}'.format(min_, max_)

    return r

def polylines(ll):
    """
    ll = [
            [(lat, lon), (lat, lon), (lat, lon), ...]
            ...
         ]
    """
    for l, c in zip(ll, plt.cm.Set2(np.linspace(0, 1, len(ll)))):
        yield {
                'path': [ { 'lat': lat, 'lng': lng } for lat, lng in l ],
                'strokeColor': rgb2hex(c),
                'strokeWeight': 2,
                'strokeOpacity': 0.8,
                }

def weighted_lines(weights, endpoints, color_pos = '#00ff00', color_neg = '#ff0000', opacity = 0.5):
    min_weight = min(weights)
    max_weight = max(weights)

    for w, (from_, to) in zip(weights, endpoints):
        color = color_pos
        w_rel = 0
        if w > 0:
            w_rel = w / max_weight
        elif min_weight < 0:
            w_rel = w / min_weight
            color = color_neg

        yield {
                'path': [ { 'lat': from_[0], 'lng': from_[1] }, { 'lat': to[0], 'lng': to[1] } ],
                'strokeColor': color,
                'strokeOpacity': opacity,
                'strokeWeight': int(10.0*w_rel),
                }

def line_sets(ll, arrows = False):
    """
    ll = [
             [((lat, lon), (lat, lon)), ((lat, lon), (lat, lon))],
             ...
         ]
    """
    assert len(ll[0][0]) == 2
    assert len(ll[0][0][0]) == 2
    assert len(ll[0][0][1]) == 2
    assert type(ll[0][0][0][0]) in (float, np.float64, np.float32)
    assert type(ll[0][0][0][1]) in (float, np.float64, np.float32)
    assert type(ll[0][0][1][0]) in (float, np.float64, np.float32)
    assert type(ll[0][0][1][1]) in (float, np.float64, np.float32)

    ll = list(ll)
    for line_set, c in zip(ll, plt.cm.Set1(np.linspace(0, 1, len(ll)))):
        for (from_, to) in line_set:
            d = {
                    'path': [ { 'lat': from_[0], 'lng': from_[1] }, { 'lat': to[0], 'lng': to[1] } ],
                    'strokeColor': rgb2hex(c),
                    'strokeWeight': 4,
                    'strokeOpacity': 0.5,
                  }
            if arrows:
                d.update({ '_ARROW': True })
            yield d



def generate_gmaps(
        lines = [],
        markers = [],
        heatmap = [],
        circles = [],
        center = (52.495372, 13.461614),
        info = [],
        default_color = '#000000'
        ):

    lines = list(lines)

    ms = []
    for m in markers:
        d = m
        if isinstance(m, tuple) or isinstance(m, np.ndarray):
            # just a coordinate
            d = { 'position': { 'lat': m[0], 'lng': m[1] }, 'title': '' }
        ms.append(d)
    s_markers = ',\n'.join('new google.maps.Marker({})'.format(json.dumps(m)) for m in ms)


    ls = []
    for l in lines:
        d = {}
        for k, v in l.items():
            if k == '_ARROW':
                d['icons'] = "[{ icon: { path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW }, offset: '100%' }]"
            else:
                d[k] = json.dumps(v)

        s = '{ ' + ','.join('{}: {}'.format(k, v) for k,v in d.items()) + '}'
        ls.append(s)
    s_lines = ',\n'.join('new google.maps.Polyline({})'.format(l) for l in ls)


    r = '''
<!DOCTYPE html>
<html>
<head>
<script src="http://maps.googleapis.com/maps/api/js?libraries=visualization&sensor=true_or_false"></script>

<script>

var center = new google.maps.LatLng({}, {});

var lines = [
    {}
    ];

var markers = [
    {}
    ];

var heatmap = [
    {}
    ];

var circles = [
    {}
    ];

var customMapType = new google.maps.StyledMapType([
      {{
        stylers: [
          {{visibility: 'simplified'}},
          {{gamma: 0.9}},
          {{weight: 0.5}},
          {{saturation: 0.01}}
        ]
      }},
      {{
        elementType: 'labels',
        stylers: [{{visibility: 'off'}}]
      }}
    ], {{
      name: 'Custom Style'
  }});
var customMapTypeId = 'custom_style';

function initialize()
{{
    var mapProp = {{
        center: center,
        zoom: 13,
        mapTypeControlOptions: {{
            mapTypeIds: [google.maps.MapTypeId.ROADMAP, customMapTypeId]
            }},
        scaleControl: true
    }};

    var map = new google.maps.Map(document.getElementById("googleMap"),mapProp);
    map.mapTypes.set(customMapTypeId, customMapType);
    map.setMapTypeId(customMapTypeId);
      

    for(var i = 0; i < lines.length; i++) {{
        lines[i].setMap(map);
    }}

    for(var i = 0; i < markers.length; i++) {{
        markers[i].setMap(map);
    }}

    for(var i = 0; i < circles.length; i++) {{
        var circle = circles[i];
        circle.setMap(map);
    }}

    var hm = new google.maps.visualization.HeatmapLayer({{
        data: heatmap
    }});
    hm.setMap(map);

}}

google.maps.event.addDomListener(window, 'load', initialize);
</script>
</head>

<body>
<div id="googleMap" style="width:1000px;height:800px;"></div>

{}

</body>
</html>
'''.format(
        center[0], center[1],
        s_lines, s_markers,
        #',\n'.join('new google.maps.Polyline({})'.format(json.dumps(l)) for l in lines),
        #',\n'.join('new google.maps.Marker({})'.format(json.dumps(m)) for m in markers),
        ',\n'.join('{{location: new google.maps.LatLng({}, {}), weight: {:.8f} }}'.format(lat, lon, float(w)) for (lat, lon, w) in heatmap),
        ','.join('new google.maps.Circle({})'.format(json.dumps(c)) for c in circles),
        '<br />\n'.join(info)
        )

    return r


if __name__ == '__main__':
    f = open('/tmp/test.html', 'w')
    f.write(generate_html_bar_graph([20, -30, 50, 0, 200, 99, -70], ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']))
    f.close

