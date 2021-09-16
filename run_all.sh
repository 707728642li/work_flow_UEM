#!/bin/bash
set -e
usage="Usage: bash ${0} batch_file"
if [ $# -eq 0 ] || [ $1 = '-h' ] ; then
	echo $usage
	exit 1
fi


# split type
for st in time user ; do
	# mode type
	for mt in s t st base ; do
		bash run_batch.sh $st $mt $1
	done
done

