from collections import Counter
import os
from pathlib import Path
import re
from typing import List, Union, Dict, Tuple

import mintpy
import numpy as np
from osgeo import gdal



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

    regex = 'ID\["EPSG",[0-9]{4,5}\]\]$'
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


def get_res(tiff):
    tiff = str(tiff)
    f =  gdal.Open(tiff)
    return f.GetGeoTransform()[1] 


def get_no_data_val(pth):
    pth = str(pth)
    f = gdal.Open(str(pth))
    if f.GetRasterBand(1).DataType > 5:
        no_data_val = f.GetRasterBand(1).GetNoDataValue()
        return np.nan if no_data_val == None else f.GetRasterBand(1).GetNoDataValue()
    else:
        return 0


def get_mintpy_vmin_vmax(dataset_path: os.PathLike, bottom_percentile: float=0.0) -> Tuple[float, float]:
    """
    Takes: 
    dataset_path: path to a MintPy hdf5 dataset
    bottom_percentile: lower end of the percentile you would like to use for vmin, vmax
                       The upper end of the percentile will be symetrical with the passed lower end.
                       Passing 0.05 as the bottom_percentile will result in 1.0 - 0.5 = 0.95 being used for the high end

    Returns: vmin, vmax values coveringt the data, centered at zero
    """
    data, _ = mintpy.utils.readfile.read(dataset_path)

    vel_min = np.percentile(data, bottom_percentile) * 100
    vel_max = np.percentile(data, 1.0-bottom_percentile) * 100
    
    vmin = -np.max([np.abs(vel_min), np.abs(vel_max)])
    vmax = np.max([np.abs(vel_min), np.abs(vel_max)])  
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
