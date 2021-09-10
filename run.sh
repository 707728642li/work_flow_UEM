#!/bin/bash
set -e
usage='Usage: bash run.sh time|user s|t|st|base'

if [ $# -eq 0 ] || [ $1 = '-h' ] ; then
	echo $usage && exit 1
fi

[ $1 != 'time' ] && [ $1 != 'user' ] && echo "Error: please type in time|user for mode!" && exit 1  
[ $2 != 's' ] && [ $2 != 't' ] && [ $2 != 'st' ] && [ $2 != 'base' ] && echo 'Error: please type in s|t|st|base for model!' && exit 1
[ $2 = 's' ] || [ $2 = 't' ] && tag=_${2}_ || tag=${2}

echo Run learning mode: ${1} >&2
echo "   " model: ${2} >&2

total_n=$( cat runlist_steps.txt | grep -c '\-\-' )
echo  $total_n samples need to be done >&2
n=0

log=run_learning.${2}.${1}_split.` date | tr ': ' '-_' `.log
echo -E 'Split mode:' $1 '\nModel type:' $2 > $log 

# [ -e ${log} ] && rm ${log}

for line in $( cat runlist_steps.txt ) ; do
        n=$((n+1))
		echo ------------------------ ${n}/${total_n} ---------------------------- >> ${log}
        i=${line%%--*}
        s=${line##*--}
        echo $( date )' START: ' $( echo $i | awk -F'/' '{print $NF}' ) >> ${log}
        # already change id_data to user-split 
		python ./src/learning/${1}/new_${2}_for_sightseeing.split_by_${1}.py -f ${i} -s ${s} -t filtered split_by_${1} $tag 
        echo $( date )' FINISH: ' $( echo $i | awk -F'/' '{print $NF}' ) >> ${log}
done

