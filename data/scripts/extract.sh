#! /bin/bash
#DIR=/rds-d2/user/wjm41/hpc-work/datasets/ZINC/real
DIR=/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL
while read p; do
  name=$(echo "$p" | cut -d'.' -f 5 | cut -d'/' -f 3)
  echo $name
  #if [ -s "$name/mols.sdf" ] || ! [ -f "$name/mols.sdf" ]; then
  if ! [ -s "$name/mols.sdf" ]; then
    if ! [ -f "$name/*.gz" ]; then # no .gz files
    wget -q $p
    tar -xf $name.tar
    rm $name.tar
    cd $name
    fi 
    for f in *.gz; do
    fname=$(echo "$f" | cut -d'.' -f 1)
    if ! [ -f "$fname.sdf" ]; then # 00000.sdf doesn't exist
    tar -xzf $f
    cd $fname
    babel *.pdbqt $DIR/$name/$fname.sdf 
    rm *.pdbqt
    cd $DIR/$name
    rm -rf $fname
    fi
    done
    rm *.tar.gz
    cat *.sdf > mols.sfd
    rm *.sdf
    mv mols.sfd mols.sdf
    #python $DIR/process_ind.py $DIR/$name
    #rm mols.sdf
    cd $DIR
  fi
done <$1

