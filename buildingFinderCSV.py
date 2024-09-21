import requests
import pandas as pd
from shapely import wkt
import time
import csv

def get_location_from_wkt(wkt_point):
    # Convert WKT to latitude and longitude
    point = wkt.loads(wkt_point)
    latitude = point.y
    longitude = point.x
    return latitude, longitude

def query_nominatim(latitude, longitude):
    # Query Nominatim API
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}&polygon_geojson=1"
    headers = {
        'User-Agent': 'GettingBuildings/1.0 ' 
    }
    print(f"query {url}")
    response = requests.get(url, headers=headers)
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
        return None

def get_osm_url(nominatim_data):
    if 'osm_type' in nominatim_data and 'osm_id' in nominatim_data:
        osm_type = nominatim_data['osm_type']
        osm_id = nominatim_data['osm_id']
        osm_url = f"https://www.openstreetmap.org/{osm_type}/{osm_id}"
        return osm_url
    else:
        return None

def process_point(wkt_point):
    try:
        latitude, longitude = get_location_from_wkt(wkt_point)
        nominatim_data = query_nominatim(latitude, longitude)
        print(nominatim_data)
        building_polygon_wkt = get_building_polygon(nominatim_data)
        osm_url = get_osm_url(nominatim_data)
        return building_polygon_wkt, osm_url
    except Exception as e:
        return None, None

def process_csv(input_csv, output_csv, point_column, delimiter=','):
    # Read the input CSV
    df = pd.read_csv(input_csv, delimiter=delimiter)

    # Initialize lists to store the results
    polygons = []
    osm_urls = []

    # Process each point in the specified column
    for point_wkt in df[point_column]:
        polygon_wkt, osm_url = process_point(point_wkt)
        polygons.append(polygon_wkt if polygon_wkt else 'Polygon not found')
        osm_urls.append(osm_url if osm_url else 'URL not found')

        # Wait for 3 second to respect rate limits
        print("WAITING FOR RATE LIMIT")
        time.sleep(3)

    # Add new columns to the DataFrame
    df['polygon'] = polygons
    df['openstreetmap_link'] = osm_urls

    df.to_csv(output_csv, index=False, sep=delimiter, quoting=csv.QUOTE_ALL)

# param to set
input_csv = 'location_with_markerGesetzt.csv'  # input CSV file
output_csv = 'output_polygons.csv'  # output CSV file
point_column = 'alternateOutline'  # Column name in the CSV that contains WKT points
delimiter = ','  # CSV delimiter (comma by default)

process_csv(input_csv, output_csv, point_column, delimiter)
