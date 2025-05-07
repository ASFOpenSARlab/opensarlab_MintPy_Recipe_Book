import base64
import csv
from datetime import datetime
import os
from pathlib import Path
import re
from typing import Union, List, Dict, Tuple
import urllib

from bokeh.plotting import figure, show, output_file, ColumnDataSource, output_notebook
from bokeh.models import LabelSet, Div
import bokeh.layouts
from pandas.core.frame import DataFrame
import h5py
import mintpy
from mintpy.utils import utils0, readfile, utils
import numpy as np
import opensarlab_lib as osl
from osgeo import gdal
from rasterio.crs import CRS
from rasterio.warp import transform_bounds, transform
from shapely.geometry import Point, box
import utm


def create_unr_gps_csv(mint_path: os.PathLike):
    mint_path = Path(mint_path)
    with osl.work_dir(mint_path):
        url = 'https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt'
        response = urllib.request.urlopen(url, timeout=5)
        content = response.read()
        rows = content.decode('utf-8').splitlines()
        holdings_txt = Path('.')/'DataHoldings.txt'
        if holdings_txt.exists():
            holdings_txt.unlink()

    with open(f'{mint_path}/GPS_stations.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', escapechar=',', quoting=csv.QUOTE_NONE)
        for row in rows:
            csv_writer.writerow([re.sub('\s+', ' ', row)])


def convert_long(long: float) -> float:
    if long > 180.0:
        long = long - 360.0
    return long

def get_utm_zone(lat: float, lon: float):
    if lat < -80 or lat > 84:
        raise ValueError("Latitude is out of UTM zone bounds.")
    
    zone = np.floor((lon + 180) / 6) + 1
    return zone

def get_gps_stations(mint_path: Union[str, os.PathLike], filename='GPS_stations.csv') -> List[str]:
    """
    Takes a path to a MintPy directory and the filename of a GPS station CSV stored within it.
    The CSV is created from https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt

    Filters the spreadsheet for GPS stations within the spatial and temporal dimensions
    of the time series. Stations on no-data pixels are removed.

    Returns a list of GPS station names
    """
    mint_path = Path(mint_path)
    
    # get the InSAR stack's corner coordinates
    geo_path = mint_path / 'inputs/geometryGeo.h5'
    atr = readfile.read_attribute(geo_path)
    bbox = box(float(atr['LON_REF2']),
               float(atr['LAT_REF3']),
               float(atr['LON_REF1']),
               float(atr['LAT_REF1']))

    # Get the start and end dates of the time series
    demErr_path = list(mint_path.glob("timeseries*_demErr.h5"))[0]
    info = gdal.Info(str(demErr_path), format='json')
    ts_start = info['metadata']['']['START_DATE']
    ts_start = datetime.strptime(ts_start, '%Y%m%d')
    ts_end = info['metadata']['']['END_DATE']
    ts_end = datetime.strptime(ts_end, '%Y%m%d')
    
    # find all stations that have data within the ts time range
    gps_stations = list()
    with open(f'{mint_path}/{filename}', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in list(csv_reader)[1:]:
            begin_date = datetime.strptime(row[7], '%Y-%m-%d')
            mod_date = datetime.strptime(row[9], '%Y-%m-%d')
            gps_lat = float(row[1])
            gps_lon = convert_long(float(row[2]))
            
            if "UTM_ZONE" in atr.keys():
                try:
                    gps_lat, gps_lon = utils.latlon2utm(atr, gps_lat, gps_lon)
                except utm.OutOfRangeError:
                    # latitude must be between 80 deg S and 84 deg N
                    continue

            gps_point = Point(gps_lon, gps_lat)

            # lat must be between 80 deg S and 84 deg N
            if "UTM_ZONE" not in atr.keys() and (gps_lat < -80.0 or gps_lat > 84.0):
                continue

            in_aoi = gps_point.within(bbox)
            in_date_range = ts_start >= begin_date and ts_end <= mod_date


            # filter GPS stations
            if in_aoi and in_date_range:
                coord = utils.coordinate(atr, lookup_file=geo_path)               
                y, x = utils.coordinate.lalo2yx(coord, gps_lat, gps_lon)

                vel, _ = readfile.read(f"{mint_path}/velocity.h5", datasetName='velocity')
                yx_vel = (vel[y][x])
    
                # remove gps stations in no-data areas of raster
                if yx_vel not in [0.0, np.nan]:
                    gps_stations.append(row[0].strip())
    
    # There must be at least 2 GPS stations in your AOI
    gps = len(gps_stations) > 1
    if not gps:
        print("There were fewer than 2 GPS sites found in your AOI")
        return gps_stations
    else:
        return gps_stations


def get_gps_dict(mint_path: os.PathLike, stations: List[str], filename='GPS_stations.csv') -> Dict:
    """
    Takes a path to a MintPy time series directory, a list of GPS station names, and the name of the GPS station
    CSV stored in the MintPy directory. 

    Returns a dictionary containing information related to the GPS stations on the stations list
    """
    mint_path = Path(mint_path)
    gps_dict = {}
    with open(f'{mint_path}/GPS_stations.csv', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in list(csv_reader)[1:]:
            if row[0] in stations:
                gps_dict[row[0]] = {
                    'lat': row[1],
                    'long': row[2],
                    'height':row[3],
                    'x': row[4],
                    'y': row[5],
                    'z': row[6],
                    'date_beg': row[7],
                    'date_end': row[8],
                    'date_mod': row[9],
                    'num_sol': row[10],
                    'st_og_name': 'na'
                }
                if len(row) > 11:
                     gps_dict[row[0]]['st_og_name'] = row[11]
    return gps_dict


def get_xyxy_web_bounds(mint_path: os.PathLike) -> Tuple[float, float, float, float]:
    """
    Takes a path to a MintPy time series directory

    Returns a tuple of corner coordinates in web-mercator projection
    (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
    """
    _, my_dict = readfile.read(f'{mint_path}/inputs/geometryGeo.h5', datasetName='height')

    x_first = float(my_dict['X_FIRST'])
    x_step = float(my_dict['X_STEP'])
    width = float(my_dict['WIDTH'])

    y_first = float(my_dict['Y_FIRST'])
    y_step = float(my_dict['Y_STEP'])
    height = float(my_dict['LENGTH'])

    # (xmin, ymin, xmax, ymax)
    bounds = (x_first, y_first+(y_step*height), x_first+(x_step*width),  y_first)
    epsg = int(my_dict['EPSG'])

    # convert bounds to web-mercator (epsg:3857)
    return transform_bounds(epsg, 3857, *bounds)

def gps_station_info_plot(mint_path, velocity_png_pth, gps_stations, gps_dict):

    xmin, ymin, xmax, ymax = get_xyxy_web_bounds(mint_path)

    longs = [convert_long(float(gps_dict[k]['long'])) for k in gps_stations]
    lats = [float(gps_dict[k]['lat']) for k in gps_stations]
    xy_web = transform(4326, 3857, longs, lats)
    
    source = DataFrame(
        data=dict(
            x=xy_web[0],
            y=xy_web[1],
            stations=gps_dict.keys(),
            lats=[gps_dict[k]['lat'] for k in gps_dict],
            longs=[convert_long(float(gps_dict[k]['long'])) for k in gps_dict],
            exes=[gps_dict[k]['x'] for k in gps_dict],
            whys=[gps_dict[k]['y'] for k in gps_dict],
            zees=[gps_dict[k]['z'] for k in gps_dict],
            heights=[gps_dict[k]['height'] for k in gps_dict],
            start_dates=[gps_dict[k]['date_beg'] for k in gps_dict],
            end_dates=[gps_dict[k]['date_end'] for k in gps_dict],
            mod_dates=[gps_dict[k]['date_mod'] for k in gps_dict],
            num_sols=[gps_dict[k]['num_sol'] for k in gps_dict],
            og_names=[gps_dict[k]['st_og_name'] for k in gps_dict],
        )
    )

    output_notebook()

    labels = LabelSet(
                x='x',
                y='y',
                text='stations',
                level='glyph',
                x_offset=-15, 
                y_offset=15, 
                source=ColumnDataSource(source))
    
    TOOLTIPS = [
        ("station", "@stations"),
        ("lat(deg)", "@lats"),
        ("long(deg)", '@longs'),
        ('height(m)', '@heights'), 
        ("x(m)", '@exes'),
        ("y(m)", "@whys"),
        ("z(m)", "@zees"),
        ("start date", "@start_dates"),
        ("end date", "@end_dates"),
        ("modification date", "@mod_dates"),
        ("NumSol", "@num_sols"),
        ("original station name", "@og_names")
    
    ]
    
    # range bounds supplied in web mercator coordinates
    p = figure(x_range=(xmin, xmax), y_range=(ymin, ymax),
               x_axis_type="mercator", y_axis_type="mercator", tooltips=TOOLTIPS)
    
    p.add_tile('CARTODBPOSITRON')
    p.scatter(marker='circle_dot', x='x', y='y', size=20, fill_alpha=0.2, color='red', alpha=0.6, source=source)
    
    p.add_layout(labels)
    
    p.title = "GPS Site Locations"
    
    with open(velocity_png_pth, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')
    
    
    data_url = f'data:image/png;base64,{encoded_image}'
    div = Div(text=f'<img src="{data_url}" alt="velocity png" width="600px"/>')
    layout = bokeh.layouts.row(p, div)
    
    show(layout)
