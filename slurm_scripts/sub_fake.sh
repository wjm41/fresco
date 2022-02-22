#! /bin/bash
for i in {0..112}; do
if [ $i -lt 10 ]; then
echo 00$i
qsub subm_extract fake00$i
elif [ $i -lt 100 ]; then
echo 0$i
qsub subm_extract fake0$i
else 
echo $i
qsub subm_extract fake$i
fi
done

