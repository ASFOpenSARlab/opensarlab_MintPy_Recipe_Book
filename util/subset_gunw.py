from pathlib import Path

import h5py
import numpy as np
from pyproj import CRS, Transformer
from shapely import from_wkt
from shapely.ops import transform


HDF5_DIMENSION_SCALE_ATTRS = {
    "DIMENSION_LIST",
    "REFERENCE_LIST",
    "CLASS",
    "NAME",
}


def subset_gunw(src_path, dst_path, aoi_4326_wkt):
    """
    Spatially subset a NISAR GUNW.

    - Preserves HDF5 hierarchy and metadata
    - Subsets all geocoded grids
    - Rebuilds HDF5 dimension scale relationships 
      (for GUNWs, this really only refers to spatial coordinates)
    - Updates product bounding polygon
    - Writes subset GUNW to dst_path

    Parameters
    ----------
    src_path : str or posix path of input GUNW
    dst_path : str or posix path of output (subset) GUNW
    aoi_4326_wkt : str WKT AOI polygon (rectangle only) in EPSG:4326.

    This script was developed with GPT-5.6 Sol assistance.
    The code was iteratively tested, reviewed, and refactored by the developer.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    aoi_4326 = from_wkt(aoi_4326_wkt)

    with h5py.File(src_path, "r") as src:
        # Find all spatial grids and calculate their subset windows
        grids = {}

        def find_grids(name, obj):
            if not isinstance(obj, h5py.Group):
                return

            if not {
                "xCoordinates",
                "yCoordinates",
                "projection",
            }.issubset(obj.keys()):
                return

            x_ds = obj["xCoordinates"]
            y_ds = obj["yCoordinates"]
            projection = obj["projection"]

            x = x_ds[:]
            y = y_ds[:]

            if "epsg_code" in projection.attrs:
                epsg = int(projection.attrs["epsg_code"])
            else:
                epsg = int(projection[()])

            transformer = Transformer.from_crs(
                "EPSG:4326",
                CRS.from_epsg(epsg),
                always_xy=True,
            )

            aoi_projected = transform(
                transformer.transform,
                aoi_4326,
            )

            xmin, ymin, xmax, ymax = aoi_projected.bounds

            xi = np.where((x >= xmin) & (x <= xmax))[0]
            yi = np.where((y >= ymin) & (y <= ymax))[0]

            if len(xi) == 0 or len(yi) == 0:
                return

            xslice = slice(xi.min(), xi.max() + 1)
            yslice = slice(yi.min(), yi.max() + 1)

            grids[obj.name] = {
                "x_path": x_ds.name,
                "y_path": y_ds.name,
                "xslice": xslice,
                "yslice": yslice,
            }

        src.visititems(find_grids)

        if not grids:
            raise ValueError("AOI does not intersect any GUNW spatial grids")

        print(f"Found {len(grids)} spatial grids:")

        for path, grid in grids.items():
            x = src[grid["x_path"]]
            y = src[grid["y_path"]]

            nx = grid["xslice"].stop - grid["xslice"].start
            ny = grid["yslice"].stop - grid["yslice"].start

            print(
                f"  {path}: "
                f"{len(y)} x {len(x)} -> "
                f"{ny} x {nx}"
            )

        # Create dict mapping dimension scales to their corresponding subset slices
        scale_slices = {}

        for grid in grids.values():
            scale_slices[grid["x_path"]] = grid["xslice"]
            scale_slices[grid["y_path"]] = grid["yslice"]

        # Determine subset slices for data arrays and their dimension-scale datasets
        def get_slices(ds):
            slices = [slice(None)] * ds.ndim

            # Determine slices for subsetting dataset dimensions
            for axis, dim in enumerate(ds.dims):
                for scale in dim.values():
                    if scale.name in scale_slices:
                        slices[axis] = scale_slices[scale.name]
                        break

            # Determine slices for subsetting dimension-scale datasets 
            if ds.name in scale_slices:
                slices[0] = scale_slices[ds.name]

            return tuple(slices)

        ##### Build the subset H5 #####
        with h5py.File(dst_path, "w") as dst:

            # Root attributes
            for key, value in src.attrs.items():
                if key not in HDF5_DIMENSION_SCALE_ATTRS:
                    dst.attrs[key] = value

            ### Recreate groups and datasets ###
            def copy_group(src_group, dst_group):

                for key, value in src_group.attrs.items():
                    if key not in HDF5_DIMENSION_SCALE_ATTRS:
                        dst_group.attrs[key] = value

                for name, obj in src_group.items():

                    # Groups
                    if isinstance(obj, h5py.Group):
                        new_group = dst_group.create_group(name)

                        copy_group(
                            obj,
                            new_group,
                        )
                        continue

                    # Named datatypes and other non-dataset objects
                    if not isinstance(obj, h5py.Dataset):
                        src_group.copy(
                            obj,
                            dst_group,
                            name=name,
                        )
                        continue

                    # Determine spatial slicing from dimension scales 
                    slices = get_slices(obj)
                    data = obj[slices]

                    # Preserve dataset storage properties
                    kwargs = {}

                    if obj.compression is not None:
                        kwargs["compression"] = obj.compression

                    if obj.compression_opts is not None:
                        kwargs["compression_opts"] = obj.compression_opts

                    if obj.shuffle:
                        kwargs["shuffle"] = obj.shuffle

                    if obj.fletcher32:
                        kwargs["fletcher32"] = obj.fletcher32

                    if obj.fillvalue is not None:
                        kwargs["fillvalue"] = obj.fillvalue

                    if obj.chunks is not None and data.ndim:
                        kwargs["chunks"] = tuple(
                            min(chunk, size)
                            for chunk, size in zip(
                                obj.chunks,
                                data.shape,
                            )
                        )

                    new = dst_group.create_dataset(
                        name,
                        data=data,
                        dtype=obj.dtype,
                        **kwargs,
                    )

                    # Copy attributes not related to dimension scales
                    for key, value in obj.attrs.items():
                        if key not in HDF5_DIMENSION_SCALE_ATTRS:
                            new.attrs[key] = value

            copy_group(src, dst)

            ### Update boundingPolygon ###
            bounding_polygon_path = (
                "/science/LSAR/identification/boundingPolygon"
            )

            if bounding_polygon_path in dst:

                original = src[bounding_polygon_path][()]

                if isinstance(original, bytes):
                    original = original.decode()

                original_polygon = from_wkt(str(original))

                subset_polygon = original_polygon.intersection(
                    aoi_4326
                )

                if subset_polygon.is_empty:
                    raise ValueError(
                        "AOI does not intersect boundingPolygon"
                    )

                ds = dst[bounding_polygon_path]

                # Preserve the original string type
                if ds.dtype.kind == "S":
                    ds[()] = subset_polygon.wkt.encode()

                else:
                    ds[()] = subset_polygon.wkt

            ### Recreate HDF5 dimension scales ###
            dimension_scales = []

            def find_dimension_scales(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return

                cls = obj.attrs.get("CLASS")

                if isinstance(cls, bytes):
                    cls = cls.decode()

                if cls == "DIMENSION_SCALE":
                    dimension_scales.append(obj.name)

            src.visititems(find_dimension_scales)

            # Mark corresponding destination datasets as scales
            for path in dimension_scales:

                if path not in dst:
                    continue

                original = src[path]

                scale_name = original.attrs.get(
                    "NAME",
                    Path(path).name,
                )

                if isinstance(scale_name, bytes):
                    scale_name = scale_name.decode()

                dst[path].make_scale(str(scale_name))

            
            ### Reattach dimension scales, matching original H5 structure ###
            def attach_dimension_scales(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return

                if obj.name not in dst:
                    return

                new = dst[obj.name]

                for axis, dim in enumerate(obj.dims):

                    # Preserve dimension labels if present
                    if dim.label:
                        new.dims[axis].label = dim.label

                    for scale in dim.values():

                        if scale.name not in dst:
                            continue

                        new.dims[axis].attach_scale(
                            dst[scale.name]
                        )

            src.visititems(attach_dimension_scales)

    print(f"\nWrote {dst_path}")
    