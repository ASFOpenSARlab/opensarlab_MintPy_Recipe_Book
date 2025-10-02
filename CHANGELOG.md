# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Update util.get_mintpy_vmin_vmax()
  - Use Dask arrays for more efficient data reads
  - Add optional stack_depth_limit to reduce resource overhead when accessing very large time series
- Update plot and time series notebooks to use updated get_mintpy_vmin_vmax
- Update OutputGeoTiffs.ipynb
  - Don't load data to get metadata

## [1.0.5]

### Changed
- Bug fix to support time series validation with GNSS when data loaded in UTM
- Update calls to support MintPy name changes that moved from "gps" to "gnss"

## [1.0.4]

### Added
- Add support for optional smallBaselineApp phase deramping step

## [1.0.3]

### Changes
- Bumps Python from 3.9 to 3.11
    - Python 3.9 end-of-life is October 2025
    - hyp3_sdk no longer supports Python 3.9
- Installs MintPy from commit [4bbca8c](https://github.com/insarlab/MintPy/commit/4bbca8c531ee021e64112da0e7885959b222a652) to pull in unwrapping error correction fix
- Pip installs a brittle set of Jupyter, matplotlib, and widget-related dependencies to support interactive matplotlib widgets in JupyterLab>4 and Python>3.9

## [1.0.2]

### Changes
- 2_CDS_Access.ipynb: update .cdsapirc to use the current CDS API endpoint

##  [1.0.1]

### Added
- Add GitHub Issues and ASF support messages to every notebook
- Support subsetting with WKT or shapefile
  - new util functions to support subsetting

### Changes
- Install MintPy from https://github.com/insarlab/MintPy.git@44d5afd to pull in kmz fix
- Remove CDS service slowdown warning
- Display dates when plotting inverted time series

## [1.0.0]

### Added
- Initial release
  - This Jupyter Book contains data recipes for loading ASF HyP3 INSAR_GAMMA and INSAR_ISCE_BURST stacks into MintPy and performing Small Baseline Subset (SBAS) line-of-sight, displacement time series analyses. It also provides options for error analysis, plotting, and outputting data to GeoTiff.