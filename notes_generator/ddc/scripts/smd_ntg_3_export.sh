source smd_0_push.sh

python create_notes_data.py \
	${SMDATA_DIR}/json_filt_ntg \
	${SMDATA_DIR}/export_ntg

popd
