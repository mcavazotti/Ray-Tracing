BLOCK_X_SIZES=( 1 2 4 8 16 32 )
BLOCK_Y_SIZES=( 8 32 64 128 256 512 1024 )
REPETITIONS=10


echo "Block dimention, Iterative Time(ms), Recursive Time(ms)" > data.csv
for x in "${BLOCK_X_SIZES[@]}"
do
	for y in "${BLOCK_Y_SIZES[@]}"
	do
		TOTAL_ITER=0
		make clean
		make ARGS="-DREC_MAX_DEPTH=5 -DBLOCK_X=$x -DBLOCK_Y=$y"
		for r in `seq 1 $REPETITIONS`
		do
			TMP=`./rayTracer 1200 800 50 2>&1 >/dev/null | tail -n3 | head -1 | sed -e 's/[^0-9]//g'`
			TOTAL_ITER=$(( $TOTAL_ITER + $TMP ))
		done
		MEAN_ITER=`echo "scale=3; $TOTAL_ITER/$REPETITIONS" | bc -l`
		TOTAL_REC=0

		make clean
		make ARGS="-DRECURSIVE -DREC_MAX_DEPTH=5 -DBLOCK_X=$x -DBLOCK_Y=$y"
		for r in `seq 1 $REPETITIONS`
		do
			TMP=`./rayTracer 1200 800 50 2>&1 >/dev/null | tail -n3 | head -1 | sed -e 's/[^0-9]//g'`
			TOTAL_REC=$(( $TOTAL_REC + $TMP ))
		done
		MEAN_REC=`echo "scale=3; $TOTAL_REC/$REPETITIONS" | bc -l`

		echo "${x}x${y}, ${MEAN_ITER}, ${MEAN_REC}" >> data.csv
	done
done
