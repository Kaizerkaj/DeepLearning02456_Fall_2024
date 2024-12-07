#!/bin/sh 
list="400 500 750 1000 1250 1500 1750 2048"
a=($list)
listCount=${#a[@]}


count=0
for i in $list
do
    if [[ $count -lt $((listCount * 1 / 4)) ]]
    then
        sed "s/param/$i/g;s/gpuVal/gpua10/g" < fastModels.sh | bsub
    elif [[ $count -lt $((listCount * 2 / 4)) ]]
    then
        sed "s/param/$i/g;s/gpuVal/gpua40/g" < fastModels.sh | bsub
    elif [[ $count -lt $((listCount * 3 / 4)) ]]
    then
        sed "s/param/$i/g;s/gpuVal/gpuv100/g" < fastModels.sh | bsub
    elif [[ $count -lt $((listCount * 4 / 4)) ]]
    then
        sed "s/param/$i/g;s/gpuVal/gpua100/g" < fastModels.sh | bsub
    fi
    count=$((count + 1))
done