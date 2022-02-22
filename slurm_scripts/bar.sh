#!/bin/bash
MPI_SIZE=32
echo $1
n=0
tot=$(wc -l $1 | cut -d' ' -f 1)

while read fname; do
n=$(expr $n + 1)
percent=$(($n * 100 / $tot))
echo "Folder Number $n / $tot ($percent % done)"
ls -lh $fname/mols.sdf
#for n in {0..11}; do
#echo "Doing $n-th processing for $fname"
#mpirun -np $MPI_SIZE -ppn $MPI_SIZE python mpi_process_mac.py $fname $n # BAD 
#cat $fname/mols_cleanup_${n}_mpi_* > $fname/mols${n}_new.smi
#mv $fname/mols${n}_new.smi $fname/mols${n}.smi
#rm $fname/mols_cleanup_${n}_mpi_*
#rm $fname/pairs_cleanup_${n}_mpi_*
#done

#rm $fname/mols.sdf
done<$1
