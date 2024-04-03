#!/bin/bash
set -e
env="opensarlab_mintpy_recipe_book"
local=$1
env_prefix=$local"/envs/"$env
python_version=$(conda run -n $env python --version | cut -b 8-10)
site_packages=$env_prefix"/lib/python"$python_version"/site-packages"

######## ARIA-Tools ########

# clone the ARIA-Tools repo and build ARIA-Tools
aria=$local"/ARIA-tools"
if [ ! -d $aria ]
then
    git clone -b release-v1.1.2 https://github.com/aria-tools/ARIA-tools.git $aria
    wd=$(pwd)
    cd $aria
    conda run -n $env python $aria/setup.py build
    conda run -n $env python $aria/setup.py install
    cd $wd
fi

path=$env_prefix"/bin:"$site_packages":"$local"/ARIA-tools/tools/bin:"$local"/ARIA-tools/tools/ARIAtools:"$PATH
pythonpath=$local"/ARIA-tools/tools:"$local"/ARIA-tools/tools/ARIAtools"

conda env config vars set -n $env GDAL_HTTP_COOKIEFILE=/tmp/cookies.txt
conda env config vars set -n $env GDAL_HTTP_COOKIEJAR=/tmp/cookies.txt
conda env config vars set -n $env VSI_CACHE=YES

# clone the ARIA-tools-docs repo
aria_docs="/home/jovyan/ARIA-tools-docs"
if [ ! -d $aria_docs ]
then
    git clone -b master --depth=1 --single-branch https://github.com/aria-tools/ARIA-tools-docs.git $aria_docs
fi

# set PATH and PYTHONPATH
conda env config vars set -n $env PYTHONPATH=$pythonpath
conda env config vars set -n $env PATH=$path