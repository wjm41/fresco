#! /bin/bash
for i in {0..7}; do
if [ $i -lt 10 ]; then
echo 00$i
qsub subm_big big_fold00$i
elif [ $i -lt 100 ]; then
echo 0$i
qsub subm_big big_fold0$i
else 
echo $i
qsub subm_big big_fold$i
fi
done

