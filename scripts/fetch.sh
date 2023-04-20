#!/bin/bash

set -ex

SMDATA_DIR="$1"

FRAXTIL_URLS="https://fra.xtil.net/simfiles/data/tsunamix/III/Tsunamix%20III%20%5BSM5%5D.zip \
https://fra.xtil.net/simfiles/data/arrowarrangements/Fraxtil%27s%20Arrow%20Arrangements%20%5BSM5%5D.zip \
https://fra.xtil.net/simfiles/data/beastbeats/Fraxtil%27s%20Beast%20Beats%20%5BSM5%5D.zip \
"
ITG_URLS="https://search.stepmaniaonline.net/static/new/In%20The%20Groove%201.zip \
https://search.stepmaniaonline.net/static/new/In%20The%20Groove%202.zip \
"

mkdir -p $SMDATA_DIR/raw
pushd $SMDATA_DIR/raw

mkdir -p "fraxtil"
pushd "fraxtil"
for I in $FRAXTIL_URLS
do
  curl $I -o temp.zip
  unzip -q -n temp.zip
  rm temp.zip
done
popd

mkdir -p "itg"
pushd "itg"
for I in $ITG_URLS
do
  curl $I -o temp.zip
  unzip -q -n temp.zip
  rm temp.zip
done
popd

popd
