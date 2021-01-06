REPETITIONS=10
SIZE_FACTOR=10
SAMPLING_FACTOR=5

make clean
make

rm *.txt

echo "set title \"Tempo de renderização\"" >> time.plg
echo "set ylabel \"Tempo (ms)\"" >> time.plg
echo "set xlabel \"Dimensão da imagem\"" >> time.plg
echo "set yrange [0:]" >> time.plg
echo "set terminal png size 1080,720" >> time.plg
echo "set style fill solid" >> time.plg
echo "set output \"./time.png\"" >> time.plg
echo "set style data histograms" >> time.plg
echo "plot '-' using 2:xtic(1) title ''" >> time.plg

for s in $(seq 1 $SIZE_FACTOR)
do
    DATA=""
    for smp in $(seq 1 $SAMPLING_FACTOR)
    do
        TOTAL=0
        for r in $(seq 1 $REPETITIONS)
        do
            #echo "$r/$REPETITIONS"
            TMP=$(./rayTracer $(( 120 * $s )) $(( 80 * $s )) $(( 10 * $smp )) 2>&1 >/dev/null | sed '2q;d' | sed -e 's/[^0-9]//g')
            TOTAL=$(( TOTAL + TMP ))
        done
        MEAN=`echo "scale=3; $TOTAL/$REPETITIONS" | bc -l`
        DATA=$DATA+' '+$MEAN
        echo "$(( 120*$s ))x$(( 80*$s ))" $DATA
    done
done