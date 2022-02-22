#! /bin/bash
DIR=/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL

#while read p; do
#  name=$(echo "$p" | cut -d'.' -f 5 | cut -d'/' -f 3)
#  echo $name
#  if ! [ -s "$name/mols.sdf" ]; then
#    wget -q $p
#    tar -xf $name.tar
#    rm $name.tar
#    cd $name
#    for f in *.gz; do
#    fname=$(echo "$f" | cut -d'.' -f 1)
#    echo $fname
#    tar -xzf $f
#    cd $fname
#    babel *.pdbqt $DIR/$name/$fname.sdf 
#    rm *.pdbqt
#    cd $DIR/$name
#    rm -rf $fname
#    done
#    rm *.tar.gz
#    cat *.sdf > mols.sfd
#    rm *.sdf
#    mv mols.sfd mols.sdf
#    cd $DIR
#  fi
#done <$1
#sed 's/.*\///g;s/\..*//g' $1 > tmp_$1
python process_real.py tmp_$1
python score_2body.py tmp_$1
