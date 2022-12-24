#! /bin/bash
for i in *.pdb; 
do
name=$(echo ${i}| cut -d'.' -f1)
printf '\n'$name'\n'
grep 'LIG' ${i} -v | grep 'HOH' -v > ${name}_new.pdb
done
