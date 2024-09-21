import requests
from shapely.geometry import Point, Polygon
from shapely import wkt

def get_building_polygon_and_link(wkt_point):
    # Convert WKT point to shapely Point
    point = wkt.loads(wkt_point)
    lat, lon = point.y, point.x
    
    # Define a small bounding box around the point
    delta = 0.001  # approximately 100m
    bbox = (lon - delta, lat - delta, lon + delta, lat + delta)
    
    # Overpass API query to get buildings within the bounding box
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      way["building"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    );
    out geom;
    """
    
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    
    if not data['elements']:
        return None, None
    
    for element in data['elements']:
        if element['type'] == 'way':
            coords = [(node['lon'], node['lat']) for node in element['geometry']]
            polygon = Polygon(coords)
            if polygon.contains(point):
                polygon_wkt = wkt.dumps(polygon)
                way_id = element['id']
                osm_link = f"https://www.openstreetmap.org/way/{way_id}"
                return polygon_wkt, osm_link
    
    return None, None

# Example usage
wkt_point = "POINT (50.72769 7.08988)" 
building_polygon, osm_link = get_building_polygon_and_link(wkt_point)
if building_polygon and osm_link:
    print("Building Polygon WKT:", building_polygon)
    print("OpenStreetMap Link:", osm_link)
else:
    print("No building found containing the point.")

