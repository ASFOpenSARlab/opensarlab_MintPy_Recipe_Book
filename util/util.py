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


def get_mintpy_vmin_vmax(dataset_path: os.PathLike) -> Tuple[float, float]:
    """
    Takes: path to a MintPy hdf5 dataset

    Returns: vmin, vmax values coveringt the data, centered at zero
    """
    data, _ = mintpy.utils.readfile.read(dataset_path)
    vel_min = -np.min(data) * 100
    vel_max = np.max(data) * 100
    vmin = -np.max([np.abs(vel_min), np.abs(vel_max)])
    vmax = np.max([np.abs(vel_min), np.abs(vel_max)])  
    return (vmin, vmax)