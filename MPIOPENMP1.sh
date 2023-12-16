for p in 1 2; do
    for t in 4; do
        mpisubmit.pl -p $p \
            --stdout /dev/null \
            --stderr /dev/null \
            -t $t \
            ./mpiopen40
    done
done