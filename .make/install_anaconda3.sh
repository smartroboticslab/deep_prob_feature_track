#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

if [ ! -d $ROOT/.anaconda3 ]; then
  echo "==>Installing anaconda 3"
  echo $ROOT
  echo $HERE
  cd $ROOT
  curl -L https://binbin-xu.github.io//tools/install_anaconda3.sh | bash -s .
fi

