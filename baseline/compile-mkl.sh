#!bin/bash
filename=$(echo "$1" | cut -f 1 -d '.')
echo "$1 hey"
echo `dpcpp $1 -DMKL_ILP64 -qmkl=parallel -o ${filename} -O3`