#!/bin/bash
#for f in performance powersave
#do
#sudo cpufreq-set -g $f
for s in `seq 0 19` #39
do
	for i in `seq 0 9`
	do
		#taskset -c $i sudo PYTHONPATH=sandbox/ python3 examples/TREASURE_tests.py $i $f
		taskset -c 3 sudo PYTHONPATH=sandbox/ python3 examples/TREASURE_tests.py 3 turbo
	done
	sleep 2
done
#done
