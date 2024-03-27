<img src="ASF_logo.svg" alt="ASF logo" width="10%"/>

# Start Here

This Jupyter Book contains data recipes for loading ASF HyP3 INSAR_GAMMA and INSAR_ISCE_BURST stacks into MintPy and performing line-of-sight, displacement time series analyses. It also provides options for error analysis, plotting, and outputting data to GeoTiff.

## MintPy

>The Miami INsar Time-series software in PYthon (MintPy as /mɪnt paɪ/) is an open-source package for Interferometric Synthetic Aperture Radar (InSAR) time series analysis. It reads the stack of interferograms (coregistered and unwrapped) in ISCE, ARIA, FRInGE, HyP3, GMTSAR, SNAP, GAMMA or ROI_PAC format, and produces three dimensional (2D in space and 1D in time) ground surface displacement in line-of-sight direction
>
>*https://github.com/insarlab/MintPy*

<br>
<div class="alert alert-success">
<font face="Calibri" size="5"><b><font color='rgba(200,0,0,0.2)'> <u>Jupyter Book Navigation</u></font></b></font>

<font face="Calibri" size="3">For an improved Jupyter Book Experience in JupyterLab, try installing the [jupyterlab-jupyterbook-navigation](https://pypi.org/project/jupyterlab-jupyterbook-navigation/) JupyterLab extension.
</font>
</div>

## How To Use This Jupyter Book

>1. ### Install the software environment needed to run the notebooks
>
>    - Run the **Install Required Software with Conda** notebook (1_Software_Environment.ipynb)
>    - Rerun this step periodically. Updates to environment config files will not take effect unless you update or recreate your environment.
>
>1. ### Configure Climate Data Store Access (optional)
>
>    - Run the **Set Up Climate Data Store Access** notebook (2_CDS_Access.ipynb)
>    - Configure CDS access if you will perform tropospheric correction
>    - If you do not wish to perform tropospheric correction, you must set the following config option: `mintpy.troposphericDelay.method = no`
>
>1. ### Access HyP3 Data
>
>    - Run the **Access & Subset HyP3 SBAS Stack (InSAR or Burst-InSAR)** notebook (3_Access_HyP3_Data.ipynb)
>    - How-to: [order interferograms from HyP3](https://storymaps.arcgis.com/stories/68a8a3253900411185ae9eb6bb5283d3)
>
>1. ### Load Data with MintPy
>
>    - Run the **A. Load HyP3 SBAS Stack into MintPy** notebook (a_Load_HyP3_Data.ipynb)
>    - Run once per SBAS stack
>  
>1. ### Configure a Time Series Analysis
>
>    - Run the **B. Configure (or Reconfigure) MintPy Time Series Analysis** notebook (b_Update_Configuration.ipynb)
>    - Run anytime you wish to update the configuration of your time series
>  
>1. ### Perform the Time Series Analysis
>
>    - Run the **C. Perform MintPy Time Series Analysis** notebook (c_MintPy_Time_Series.ipynb)
>
>1. ### Run Post-Time Series Workflows
>
>    - Error Analysis (Error_Analysis.ipynb)
>    - Plotting (Plots.ipynb)
>    - Output Results to GeoTiff (Output_GeoTiff.ipynb)
>  
> 1. ### Update Your Configuration and Reprocess the Time Series
>
>    - Use the results of your time series and error analyses to make configuration adjustments and reprocess the time series by repeating steps 5 and 6.


## Practical Notes About Using MintPy
MintPy loads data from any source or processor into two HDF5 datasets:
- `geometryGeo.h5`
- `ifgramStack.h5`

As you run steps in the time series script, `smallbaselineapp.py`, additional HDF5 data sets will be created, which become inputs for following steps.

**You only need to load your data once**
- If you have loaded your times series from a bunch of large GeoTiffs, you can delete them to conserve space after loading your data.
- You will be able to reconfigure and rerun your time series without needing to reload your data.
- If you reconfigure your time series, you must rerun the time series for the updates to take effect.
- Once you have run your time series, until you wish to reconfigure it, you can rerun any data recipes in the `Extras` section without having to rerun the time series. 





