REPETITIONS=10
SIZE_FACTOR=10
SAMPLING_FACTOR=5

make clean
make

rm *.txt

for smp in $(seq 1 $SAMPLING_FACTOR)
do
    for s in $(seq 1 $SIZE_FACTOR)
    do
        TOTAL=0
        for r in $(seq 1 $REPETITIONS)
        do
            #echo "$r/$REPETITIONS"
            TMP=$(./rayTracer $(( 120 * $s )) $(( 80 * $s )) $(( 10* $smp )) 2>&1 >/dev/null | sed '2q;d' | sed -e 's/[^0-9]//g')
            TOTAL=$(( TOTAL + TMP ))
        done
        MEAN=`echo "scale=3; $TOTAL/$REPETITIONS" | bc -l`
        echo "$(( 120*$s ))x$(( 80*$s ))" $MEAN
    done
done