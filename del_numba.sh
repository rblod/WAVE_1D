#!/bin/bash

for file in $(ls *py)
do
	PAT1=$(grep "from numba import jit" $file)
	PAT2=$(grep "@jit" $file)
	if [ ${#PAT1} -gt 0 ]; then
		sed -e "s/$PAT1/#$PAT1/g" -e "s/$PAT2/#$PAT2/g" $file > $file.tmp
		\mv $file.tmp $file
	fi	
done	