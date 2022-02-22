#! /bin/bash
MPI_SIZE=8
echo $1
n_fold=0
tot=$(wc -l $1 | cut -d' ' -f 1)

while read fname; do
n_fold=$(expr $n_fold + 1)
percent=$(($n_fold * 100 / $tot))
echo "Folder Number $n_fold / $tot ($percent % done)"

if ! [ -f $fname/done.txt ] # touch.txt doesn't exist
then
echo "$fname needs processing"
#for n in {0..11}; do
#echo "Doing $n-th processing for $fname"
#mpirun -np $MPI_SIZE -ppn $MPI_SIZE python mpi_process_mac.py $fname $n 12 > /dev/null < /dev/null
#cat $fname/mols_cleanup_${n}_mpi_* > $fname/mols${n}_new.smi
#mv $fname/mols${n}_new.smi $fname/mols${n}.smi
#rm $fname/mols_cleanup_${n}_mpi_*
#rm $fname/pairs_cleanup_${n}_mpi_*
#done

else
echo "$fname done!"
fi
#rm $fname/mols.sdf
done<$1


