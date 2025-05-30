from collections import Counter
from datetime import datetime
import os
from pathlib import Path
import re
from typing import List, Union, Dict, Tuple, Optional

import geopandas as gpd
from mintpy.utils import readfile
import numpy as np
from osgeo import gdal, ogr, osr
gdal.UseExceptions()
from pyproj import Transformer
import rasterio
from shapely.geometry import Polygon
from shapely.ops import transform
import shapely.wkt


def get_projection(img_path: Union[Path, str]) -> Union[str, None]:
    """
    Takes: a string or posix path to a product in a UTM projection

    Returns: the projection (as a string) or None if none found
    """
    img_path = str(img_path)
    try:
        info = gdal.Info(img_path, format='json')['coordinateSystem']['wkt']
    except KeyError:
        return None
    except TypeError:
        raise FileNotFoundError

    regex = r'ID\["EPSG",[0-9]{4,5}\]\]$'
    results = re.search(regex, info)
    if results:
        return results.group(0).split(',')[1][:-2]
    else:
        return None


def get_projections(tiff_paths: List[Union[os.PathLike, str]]) -> Dict:
    """
    Takes: List of string or posix paths to geotiffs
    
    Returns: Dictionary key: epsg, value: number of tiffs in that epsg 
    """
    epsgs = []
    for p in tiff_paths:
        epsgs.append(get_projection(p))

    epsgs = dict(Counter(epsgs))
    return epsgs


def get_res(tiff: os.PathLike) -> float:
    """
    Takes: path to a GeoTiff

    Returns: The GeoTiff's resolution
    """
    tiff = str(tiff)
    f =  gdal.Open(tiff)
    return f.GetGeoTransform()[1] 


def get_no_data_val(pth: os.PathLike) -> Union[None, float, int]:
    """
    Takes: path to a GeoTiff

    Returns: The GeoTiff's no-data value
    """
    pth = str(pth)
    f = gdal.Open(str(pth))
    if f.GetRasterBand(1).DataType > 5:
        no_data_val = f.GetRasterBand(1).GetNoDataValue()
        return np.nan if no_data_val == None else f.GetRasterBand(1).GetNoDataValue()
    else:
        return 0


def get_mintpy_vmin_vmax(dataset_path: os.PathLike, mask_path: os.PathLike=None, bottom_percentile: float=0.0) -> Tuple[float, float]:
    """
    Takes: 
    dataset_path: path to a MintPy hdf5 dataset
    mask_path: path to a MintPy hdf5 dataset containing a coherence mask (such as 'maskTempCoh.h5')
    bottom_percentile: lower end of the percentile you would like to use for vmin, vmax
                       The upper end of the percentile will be symetrical with the passed lower end.
                       Passing 0.05 as the bottom_percentile will result in 1.0 - 0.05 = 0.95 being used for the high end

    Returns: vmin, vmax values covering the data (or masked data), centered at zero
    """
    data, _ = readfile.read(dataset_path)

    if mask_path:
        mask, _ = readfile.read(mask_path)
        data *= mask

    vel_min = np.nanpercentile(data, bottom_percentile) * 100
    vel_max = np.nanpercentile(data, 1.0-bottom_percentile) * 100
    
    vmin = -np.nanmax([np.abs(vel_min), np.abs(vel_max)])
    vmax = np.nanmax([np.abs(vel_min), np.abs(vel_max)])  
    return (vmin, vmax)

def get_recent_mintpy_config_path() -> Union[os.PathLike, None]:
    """
    Returns the mintpy config path saved in .recent_mintpy_config
    if it exists else None
    """
    recent_mintpy_path = Path.cwd() / '.recent_mintpy_config'

    try:
        with open(recent_mintpy_path, 'r') as f:
            line = f.readline()    
            if Path(line).exists():
                return Path(line)
    except FileNotFoundError:
        return None
    except:
        raise
    return None # Found a recent path but it no longer exists


def write_recent_mintpy_config_path(pth: Union[str, os.PathLike]):
    recent_mintpy_path = Path.cwd() / '.recent_mintpy_config'
    with open(recent_mintpy_path, 'w+') as f:
        f.write(str(pth))


def get_epsg(geotiff_path: Union[str, os.PathLike]) -> str:
    """
    Takes: A string path or posix path to a GeoTiff

    Returns: The string EPSG of the Geotiff
    """
    ds = gdal.Open(str(geotiff_path))
    proj = ds.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    srs.AutoIdentifyEPSG()
    return srs.GetAuthorityCode(None)
    

def get_geotiff_bbox(geotiff_path: Union[str, os.PathLike], dst_epsg: str=None) -> Polygon:
    """
    Takes:
    geotiff_path: path to a GeoTiff
    dst_epsg: optional EPSG for reprojection

    Returns: The GeoTiffs bounding box as a shapely.geometry.Polygon
    """
    with rasterio.open(geotiff_path) as dataset:
        bounds = dataset.bounds
        min_x, min_y = (bounds.left, bounds.bottom)
        max_x, max_y = (bounds.right, bounds.top)
        
    if dst_epsg:
        srs_crs = dataset.crs
        transformer = Transformer.from_crs(srs_crs, f'EPSG:{str(dst_epsg)}', always_xy=True)
        min_x, min_y = transformer.transform(bounds.left, bounds.bottom)
        max_x, max_y = transformer.transform(bounds.right, bounds.top)

    return Polygon([
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (min_x, min_y)
    ])


def possible_wgs84_wkt(wkt: str) -> bool:
    """
    If WKT Polygon falls within the range of valid WGS84 coords,
    prompts user to indicate whether the WKT is WGS84 or UTM

    Takes: Well-Known-Text polygon string

    Returns: True if the WKT could be WGS84 else False
    """
    lon_regex = r"(?:\(|,)(-?\d{1,6}\.?\d{0,6})"
    lat_regex = r"(?<=\s)-?\d{,6}.?\d{,6}"
    
    lon_results = re.findall(lon_regex, wkt)
    lon_results = [float(n) for n in lon_results]
    lat_results = re.findall(lat_regex, wkt)
    lat_results = [float(n) for n in lat_results]
    if -180.0 <= np.min(lon_results) and np.max(lon_results) <= 180.0 \
    and -90.0 <= np.min(lat_results) and np.max(lat_results) <= 90.0:
        while True:
            print("Detected possible WGS84 (lat/lon) coordinates")
            wgs84 = input("Are these lat/lon coordinates? (y or n)")
            if wgs84 in ["y", "n"]:
                wgs84 = True if wgs84 == 'y' else False
                return wgs84
    else:
        return False


def project_wkt_polygon(wkt_polygon: str, source_epsg: Union[int, str], target_epsg: Union[int, str]) -> str:
    """
    Takes: 
    wkt_polygon: A Well-Known-Text POLYGON string
    source_epsg: wkt_polygon's EPSG
    target_epsg: the target EPSG for projection

    Returns: A Well-Known-Text string in the target EPSG
    """
    polygon = shapely.wkt.loads(wkt_polygon)
    transformer = Transformer.from_crs(f"EPSG:{source_epsg}", f"EPSG:{target_epsg}", always_xy=True)
    transformed_polygon = transform(transformer.transform, polygon)
    return transformed_polygon.wkt


def get_valid_wkt() -> Tuple[str, Polygon]:
    """
    Prompts user for WKT

    Returns: WKT string, Shapely Polygon from WKT
    """
    
    while True:
        try:
            wkt = input('Please enter your WKT (e.g. "POLYGON((-148.4241 64.6077,-146.9478 64.6077,-146.9478 65.1052,-148.4241 65.1052,-148.4241 64.6077))": ')

            shapely_geom = shapely.wkt.loads(wkt)
            
            if not gpd.GeoSeries([shapely_geom]).is_valid[0]:
                print('Invalid geometry detected. Please enter a valid WKT.')
                continue
            
            return wkt, shapely_geom
        except Exception as e:
            print(f'Error: {e}. Please enter a valid WKT.')


def check_within_bounds(wkt_shapely_geom: Polygon, gdf: gpd.GeoDataFrame) -> bool:
    """
    wkt_shapely_geom: A shapely Polygon describing a subset AOI
    gdf: a geopandas.GeoDataFrame containing geometries for each dataset to subset to wkt_shapely_geom

    returns: True if wkt_shapely_geom is contained within all geometries in the GeoDataFrame, else False
    """   
    return all(wkt_shapely_geom.within(geom) for geom in gdf['geometry'])


def save_shapefile(
    ogr_geom: ogr.Geometry, 
    epsg: Union[str, int], 
    dst_path: Optional[Union[str, os.PathLike]]=Path.cwd()/f'shape_{datetime.strftime(datetime.now(), "%Y%m%dT%H%M%S")}.shp'
) -> None:
    """
    Writes a shapefile from an ogr geometry in a given projection
    
    ogr_geom: An ogr geometry
    epsg: the EPSG projection to apply to the shapefile
    dst_path: (optional) shapefile destination path
    """
    epsg = int(epsg)
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(str(dst_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    layer = ds.CreateLayer('', srs, ogr.wkbPolygon)
    defn = layer.GetLayerDefn()

    feat = ogr.Feature(defn)
    feat.SetGeometry(ogr_geom)
    
    layer.CreateFeature(feat)
    feat = geom = None

    ds = layer = feat = geom = None

