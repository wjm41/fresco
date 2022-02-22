#! /bin/bash

> bad_dirs.txt
while read fold; do
for i in {0..11}; do
if [ $i -lt 11 ]; then
j=$((i+1))
fi
if ! [ -s $fold/mols$i.pickle ] && ! [ -s $fold/pairs_mpi_$i.pickle ]; then
echo $fold >> bad_dirs.txt
ls -lh $fold
break
fi
if [ -s $fold/mols$i.pickle ] && ! [ -s $fold/mols$j.pickle ]; then
echo $fold >> bad_dirs.txt
ls -lh $fold
break
fi
done
done<bigs.txt
