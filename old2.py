import requests
from shapely import wkt

def get_location_from_wkt(wkt_point):
    # Convert WKT to latitude and longitude
    point = wkt.loads(wkt_point)
    latitude = point.y
    longitude = point.x
    return latitude, longitude

def query_nominatim(latitude, longitude):
    # Query Nominatim API
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}&polygon_geojson=1"
    print(f"query {url}")
    response = requests.get(url)
    print(f"response {response}")
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Error querying Nominatim API")

def get_building_polygon(nominatim_data):
    if 'geojson' in nominatim_data and nominatim_data['geojson']['type'] == 'Polygon':
        # Get the coordinates of the polygon
        coordinates = nominatim_data['geojson']['coordinates'][0]
        polygon_wkt = f"POLYGON(({', '.join([f'{coord[0]} {coord[1]}' for coord in coordinates])}))"
        return polygon_wkt
    else:
        raise Exception("No building polygon found in Nominatim data")

def get_osm_url(nominatim_data):
    if 'osm_type' in nominatim_data and 'osm_id' in nominatim_data:
        osm_type = nominatim_data['osm_type']
        osm_id = nominatim_data['osm_id']
        osm_url = f"https://www.openstreetmap.org/way/{osm_id}"
        return osm_url
    else:
        raise Exception("No OSM URL found in Nominatim data")

def main(wkt_point):
    latitude, longitude = get_location_from_wkt(wkt_point)
    nominatim_data = query_nominatim(latitude, longitude)
    building_polygon_wkt = get_building_polygon(nominatim_data)
    osm_url = get_osm_url(nominatim_data)
    return building_polygon_wkt, osm_url

# Example usage
wkt_point = "POINT(7.08991 50.72773)"  # Example WKT point (longitude, latitude)
try:
    building_polygon_wkt, osm_url = main(wkt_point)
    print("Building Polygon WKT:", building_polygon_wkt)
    print("OpenStreetMap URL:", osm_url)
except Exception as e:
    print(str(e))
