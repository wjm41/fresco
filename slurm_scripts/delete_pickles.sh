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

while read fold; do
echo $fold
rm $fold/*.pickle # delete all the bad pickles
rm $fold/*.npy
rm $fold/*.csv
rm $fold/*.smi
#rm $dir/$fold/mols.sdf
#rm $dir/$fold/pairs_clean*
done < $1
