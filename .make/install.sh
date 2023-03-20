#!/bin/bash

set -e

echo_bold () {
  echo -e "\033[1m$*\033[0m"
}

echo_warning () {
  echo -e "\033[33m$*\033[0m"
}

conda_check_installed () {
  if [ ! $# -eq 1 ]; then
    echo "usage: $0 PACKAGE_NAME"
    return 1
  fi
  conda list | awk '{print $1}' | egrep "^$1$" &>/dev/null
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

cd $ROOT

source .anaconda3/bin/activate

# ---------------------------------------------------------------------------------------

echo_bold "==> Installing the right pip and dependencies for the fresh python"
pip install --upgrade pip 
conda install python=3.6  # meet tensorflow requirements
conda install ipython

#echo_bold "==> Installing computer vision-related packages"
#pip install \
#  jupyter \
#  cython\
#  numpy\
#  matplotlib\
#  opencv-python \
#  opencv-contrib-python \
#  plyfile \
#  pandas \
#  requests \
#  scipy \
#  imageio \
#  scikit-image \
#  sklearn \
#  pyyaml \
#  tqdm \
#  transforms3d \
#
#echo_bold "==> Installing deep learning-related packages"
#pip install future 
#conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
#pip install tensorboard

echo_bold "==> Installing requirements"
# pip install -r setup/requirements.txt
conda env update --file environment30X.yml
# pip install -e .

# ---------------------------------------------------------------------------------------

echo_bold "\nAll is well! You can start using this!
  $ source .anaconda3/bin/activate
"
