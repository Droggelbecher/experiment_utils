#!/usr/bin/env python

import sys
from tracking_pb2 import Track, TrackEntry


def read_data(filename):
    """
    Yields TrackEntry objects
    """
    file_ = open(filename, 'rb')
    while True:
        block_length_s = file_.read(2)

        if len(block_length_s) < 2:
            break

        block_length = (ord(block_length_s[1]) << 8) | ord(block_length_s[0])
        block = file_.read(block_length)

        entry = TrackEntry()
        entry.ParseFromString(block)

        yield entry


def generate_ttp(trace):

    # For TTP spec see
    # https://confluence.tomtomgroup.com/display/POS/How+to+read+ttp+log+files

    r = 'BEGIN:ApplicationVersion=TomTom Positioning 0.2\n'
    r += '0.000,245,0,SENSOR=Location,periodMS=1000,X=0,Y=0,Z=0,oriX=0,oriY=0,oriZ=0,accuracy=1000000.000000,offset=0.000000,sensitivity=1.000000,min=-1000.000000,max=1000.000000\n'

    startdate = None
    for e in trace:
        if startdate is None:
            startdate = float(e.dateInMillis)

        l = '{},245,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
            (float(e.dateInMillis) - startdate) / 1000.0,
            0, # channel
            e.longitude,
            float(e.gpsVerticalAccuracy),
            e.latitude,
            float(e.gpsHorizontalAccuracy),
            e.gpsAltitude,
            1.0, # altitude accuracy
            e.gpsCourse, # heading
            1.0, # heading accuracy
            e.gpsSpeed,
            1.0, # speed accuracy
            1.0, # slope
            1.0, # slope accuracy
            e.distance,
            1.0, # distance accuracy
            e.dateInMillis / 1000.0,
            3, # source of data (Unknown=0, Network=1, GNSS=3, GNSS_LSQ=5)
            1 # GNSS used?
            )


        r += l
    r += 'END\n'
    return r








# ----

def read_metafile(file_):
    block = file_.read()

    entry = Track()
    entry.ParseFromString(block)

    return entry


def generate_gmaps(traces):

    trips = []

    for trace in traces:
        l = []
        for e in trace:
            try:
                l.append((e.latitude, e.longitude))
            except KeyError:
                pass
        trips.append(l)

    r = '''
<!DOCTYPE html>
<html>
<head>
<script
src="http://maps.googleapis.com/maps/api/js">
</script>

<script>

var trips = [
    {}
    ];

var colors = [
    "#000000",
    "#ff0000", "#00ff00", "#0000ff",
    "#ff00ff", "#ffff00", "#ffffff"
    ];

function initialize()
{{
    var mapProp = {{
        center: trips[0][0],
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


}}

google.maps.event.addDomListener(window, 'load', initialize);
</script>
</head>

<body>
<div id="googleMap" style="width:1000px;height:800px;"></div>
</body>
</html>
'''.format(','.join('[' + 
            ','.join(('new google.maps.LatLng({}, {})'.format(lat, lon) for (lat, lon) in trip))
            + ']' for trip in trips))

    return r

if __name__ == '__main__':

    # datafile

    traces = []

    for filename in sys.argv[1:]:
        traces.append(list(read_data(filename)))

    maps_outname = 'out_gmaps.html'
    print('writing ' + maps_outname)
    with open(maps_outname, 'w') as mapsfile:
        mapsfile.write(generate_gmaps(traces))


    ttp_outname = 'out.ttp'
    print('writing ' + ttp_outname)
    with open(ttp_outname, 'w') as ttpfile:
        ttpfile.write(generate_ttp(traces))



    #print('writing ' + data_outname)
    #with open(data_outname, 'w') as dataoutfile:
        #for entry in entries:
            #dataoutfile.write(str(entry))
            #dataoutfile.write('----------------\n')

    # metafile

    #print('writing ' + meta_outname)
    #with open(metaname, 'rb') as metafile:
        #with open(meta_outname, 'w') as metaoutfile:
            #track = read_metafile(metafile)
            #metaoutfile.write(str(track))


