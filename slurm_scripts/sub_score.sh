#! /bin/bash
for i in {0..103}; do
if [ $i -lt 10 ]; then
echo 00$i
qsub subm_score score_fold00$i
elif [ $i -lt 100 ]; then
echo 0$i
qsub subm_score score_fold0$i
else 
echo $i
qsub subm_score score_fold$i
fi
done

