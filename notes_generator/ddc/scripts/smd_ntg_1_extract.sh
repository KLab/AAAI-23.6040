source smd_0_push.sh

echo "SMDATA_DIR" $SMDATA_DIR

PACKS_DIR=${SMDATA_DIR}/raw/${1}
JSON_DIR=${SMDATA_DIR}/json_raw_ntg/${1}

mkdir -p $PACKS_DIR
mkdir -p $JSON_DIR

python extract_json_ntg.py \
	${PACKS_DIR} \
	${JSON_DIR} \
	${2}

popd
