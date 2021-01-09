make clean
make ARGS="-DREC_MAX_DEPTH=5"

REPETITIONS=10

TOTAL=0

for r in `seq 1 $REPETITIONS`
do
	TMP=`./rayTracer 1200 800 50 2>&1 >/dev/null | tail -n3 | head -1 | sed -e 's/[^0-9]//g'`
	TOTAL=$(( $TOTAL + $TMP ))
done
MEAN=`echo "scale=3; $TOTAL/$REPETITIONS" | bc -l`

NUM_THREADS=`./rayTracer 12 8 5 2>&1 >/dev/null | head -1 | sed -e 's/[^0-9]//g'`

echo "Number of threads:${NUM_THREADS} Average time: ${MEAN}ms"
