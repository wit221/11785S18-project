#!/bin/bash
num=`cat model.ver`
num=$(($num+1))
echo $num > model.ver
cp cvae.model.pt models/cvae.model.pt.$num
if [ "$?" != "0" ] ; then
    echo "Failed to create a copy of the model"
    exit 1
fi
python -u src/modeling/cvae_m_cont.py --seed 0 --cuda  -n 20 -enum parallel -zd 100 -hd 256 -lr 0.0000001 -b1 0.95 -bs 10 -log ./tmp.log

