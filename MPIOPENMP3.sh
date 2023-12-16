for p in 4; do
    for t in 1 2 4 8; do
        mpisubmit.pl -p $p \
	          --W 00:30 \
            --stdout /dev/null \
            --stderr /dev/null \
            -t $t \
            ./mpiopen160
    done
done