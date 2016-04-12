#!/usr/bin/env python

def generate_gmaps(trips = [], markers = [], heatmap = []):
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

    print("markers=", markers)

    r = '''
<!DOCTYPE html>
<html>
<head>
<script src="http://maps.googleapis.com/maps/api/js?libraries=visualization&sensor=true_or_false"></script>

<script>

var trips = [
    {}
    ];

var markers = [
    {}
    ];

var heatmap = [
    {}
    ];

var colors = [
    "#000000"
    ];

function initialize()
{{
    var mapProp = {{
        center: markers[0],
        zoom:15,
        mapTypeId:google.maps.MapTypeId.ROADMAP
    }};
      
    var map = new google.maps.Map(document.getElementById("googleMap"),mapProp);

    for(var i = 0; i < trips.length; i++) {{
        var flightPath=new google.maps.Polyline({{
          path: trips[i],
          strokeColor: colors[i % colors.length],
          strokeOpacity:0.8,
          strokeWeight:2
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
</body>
</html>
'''.format(
        ',\n'.join('['
        + ',\n  '.join(('new google.maps.LatLng({}, {})'.format(lat, lon) for (lat, lon) in trip))
        + ']' for trip in trips),

        ',\n'.join('new google.maps.LatLng({}, {})'.format(lat, lon) for (lat, lon) in markers),

        ',\n'.join('{{location: new google.maps.LatLng({}, {}), weight: {:.8f} }}'.format(lat, lon, float(w)) for (lat, lon, w) in heatmap)
        )

    return r
