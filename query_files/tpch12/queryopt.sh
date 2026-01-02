#/bin/bash

for ((i=1; i<=12; i++)); do
  # echo "python sql_optimize.py --database indexselection_tpch___1 --file bak/workload${i}.txt --output workload${i}.txt"
  python sql_optimize.py --database indexselection_tpch___1 --file bak/workload${i}.txt --output workload${i}.txt
done
