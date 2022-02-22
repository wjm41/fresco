#! /bin/bash

dir='/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL'
#while read p; do
#  name=$(echo "$p")
#  for i in {0..11}; do 
#  if [ -s "$dir/$name/pairs_mpi_$i.pickle" ] && [ -s "$dir/$name/mols$i.pickle" ]; then #BOTH pickles exist
#    echo $name
#    rm $dir/$name/mols$i.pickle
#  fi
#  done
#done <$1

#while read p; do
#  name=$(echo "$p")
#  if [ -s "$dir/$name/pairs.pickle" ] && [ -s "$dir/$name/mols.sdf" ]; then #BOTH pickles exist
#  rm $dir/$name/mols.sdf
#  fi
#done <$1

while read fname; do
#fname=$(cat $fold_name)
del="true"
#echo $fname
for i in {0..11}; do
if ! [ -s "$fname/mols${i}.smi" ] || ! [ -s "$fname/pairs_mpi_${i}.pickle" ]; then
del="false"
#cat $fname/mols_cleanup_${i}_mpi_* > $fname/mols${i}_new.smi
#mv $fname/mols${i}_new.smi $fname/mols${i}.smi
#rm $fname/mols_cleanup_${i}_mpi_*
fi
done
if [ "$del" == "true" ]; then
#if [ "$del" == "false" ]; then
echo $fname
rm $fname/mols.sdf
fi
done < $1
