for N in 4 8 16 32 ; do \
    bsub -n 1 -q normal -o /dev/null -e /dev/null -R  "affinity[core(10,same=socket,exclusive=(socket,alljobs)):membind=localonly:distribute=pack(socket=1)]"  "OMP_NUM_THREADS=$N" ./para160Sub
done