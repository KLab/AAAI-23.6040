for COLL in fraxtil itg
do
	source smd_ntg_1_extract.sh ${COLL} --itg
done

for COLL in fraxtil itg
do
	source smd_ntg_2_filter.sh ${COLL}
done

source smd_ntg_3_export.sh
