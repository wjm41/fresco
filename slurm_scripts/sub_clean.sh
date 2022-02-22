#! /bin/bash
for i in {0..4}; do
if [ $i -lt 10 ]; then
echo 00$i
#./test.sh clean_fold00$i
qsub subm_clean clean_fold00$i
elif [ $i -lt 100 ]; then
echo 0$i
#./test.sh clean_fold0$i
qsub subm_clean clean_fold0$i
else 
echo $i
#qsub subm_clean clean_fold$i
fi
done

