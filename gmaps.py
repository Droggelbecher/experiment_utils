#!/usr/bin/env python


def generate_html_bar_graph(heights, names = None):
    width = len(heights) * 24
    height = 100

    min_ = min(heights)
    max_ = max(heights)

    hs = [height*h/max_ if h > 0 else -height*h/min_ for h in heights]

    r = ''

    r += '<div style="width: {}px; height: {}px; border: 1px solid #a0a0a0">'.format(width, height)
    for h in hs:
        r += '<div style="height: {}px; width: 20px; margin: 2px; display: inline-block; position: relative; background-color: #00ff00; vertical-align: baseline;"></div>'.format(h if h > 0 else 0)
    r += '</div>'

    if min_ < 0:
        r += '<div style="width: {}px; height: {}px; border: 1px solid #a0a0a0">'.format(width, height)
        for h in hs:
            r += '<div style="height: {}px; width: 20px; margin: 2px; display: inline-block; position: relative; background-color: #ff0000; vertical-align: top;"></div>'.format(-h if h < 0 else 0)
        r += '</div>'


    if names is not None:
        r += '<div style="width: {}px; border: 1px solid #a0a0a0">'.format(width)
        for name in names:
            r += '<div style="width: 20px; margin: 2px; display: inline-block; position: relative; vertical-align: baseline;">{}</div>'.format(name)
        r += '</div>'

    r += '<br />Min: {}<br />Max: {}'.format(min_, max_)

    return r


def generate_gmaps(
        trips = [],
        trip_weights = [],
        markers = [],
        heatmap = [],
        center = (52.495372, 13.461614),
        info = []
        ):
    """
    trips: iterable over iterable of (lat, lon) pairs

    eg: trips = [
            [ (1,2), (3,4), (5,6) ],
            [ (100, 100), (101, 102), (101, 103), (102, 100) ],
            ...
        ]

    heatmap = [
        (lat, long, v),
        ...
        ]
    """

    tw_min = min(trip_weights)
    tw_max = max(trip_weights)

    trip_colors = ['#000000'] * len(trips)
    trip_strokes = [0] * len(trips)

    for i, w in enumerate(trip_weights):
        #w_rel = ((w - tw_min) / (tw_max - tw_min))
        #green = 255 if w_rel > .5 else int(255 * 2 * w_rel)
        #red = 255 if w_rel < .5 else int(255 * 2 * (1.0 - w_rel))
        #trip_colors[i] = '#{:02X}{:02X}00'.format(red, green)

        if w > 0:
            trip_colors[i] = '#00FF00'
            w_rel = w / tw_max
            trip_strokes[i] = int(10.0 * w_rel)

        elif w < 0:
            trip_colors[i] = '#FF0000'
            w_rel = w / tw_min
            trip_strokes[i] = int(10.0 * w_rel)


    r = '''
<!DOCTYPE html>
<html>
<head>
<script src="http://maps.googleapis.com/maps/api/js?libraries=visualization&sensor=true_or_false"></script>

<script>

var center = new google.maps.LatLng({}, {});

var trips = [
    {}
    ];

var markers = [
    {}
    ];

var heatmap = [
    {}
    ];

var trip_colors = [
    {}
    ];

var trip_strokes = [
    {}
    ];

function initialize()
{{
    var mapProp = {{
        center: center,
        zoom: 13,
        mapTypeId: google.maps.MapTypeId.ROADMAP
    }};
      
    var map = new google.maps.Map(document.getElementById("googleMap"),mapProp);

    for(var i = 0; i < trips.length; i++) {{
        var flightPath=new google.maps.Polyline({{
          path: trips[i],
          strokeColor: trip_colors[i],
          strokeOpacity: 1.0,
          strokeWeight: trip_strokes[i]
          }});

        flightPath.setMap(map);
    }}

    for(var i = 0; i < markers.length; i++) {{
        var marker = new google.maps.Marker({{
          position: markers[i],
          title:"x"
          }});

        marker.setMap(map);
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

Min. weight: {}<br />
Max. weight: {}<br />
{}

</body>
</html>
'''.format(
        center[0], center[1],
        ',\n'.join('['
            + ',\n  '.join(('new google.maps.LatLng({}, {})'.format(t[0], t[1]) for t in trip))
            + ']' for trip in trips),
        ',\n'.join('new google.maps.LatLng({}, {})'.format(lat, lon) for (lat, lon) in markers),
        ',\n'.join('{{location: new google.maps.LatLng({}, {}), weight: {:.8f} }}'.format(lat, lon, float(w)) for (lat, lon, w) in heatmap),
        ','.join('"{}"'.format(c) for c in trip_colors),
        ','.join('{}'.format(c) for c in trip_strokes),

        tw_min,
        tw_max,
        '<br />\n'.join(info)
        )

    return r


if __name__ == '__main__':
    f = open('/tmp/test.html', 'w')
    f.write(generate_html_bar_graph([20, -30, 50, 0, 200, 99, -70], ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']))
    f.close

