#!/bin/bash

set -ex

SMDATA_DIR="$1"

pushd notes_generator/ddc/scripts

source smd_ntg_all.sh

popd

PYTHONPATH="$(pwd)" python scripts/mel_convert.py all --mel_save_dir "$SMDATA_DIR"/train_data/mel_log \
  --wav_base_path "$SMDATA_DIR"/export_ntg/audio \
  --m_live_data_path "$SMDATA_DIR"/export_ntg/m_live_data.csv \
  --parallel

PYTHONPATH="$(pwd)" python scripts/onsets_converter.py stepmania \
  --save_path "$SMDATA_DIR"/train_data/score_onsets_1 \
  --data_path "$SMDATA_DIR"/export_ntg/notes_data.json
