#! /bin/bash
for i in {0..8}; do
if [ $i -lt 10 ]; then
echo 00$i
qsub subm_extract missing_fold00$i
#qsub subm_extract fake00$i
elif [ $i -lt 100 ]; then
echo 0$i
qsub subm_extract missing_fold0$i
#qsub subm_extract fake0$i
else 
echo $i
qsub subm_extract missing_fold$i
#qsub subm_extract fake$i
fi
done

