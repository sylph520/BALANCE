#!/bin/bash
set -e

log_fn='tpchc_testset.log'

echo ">>>>>>>>>>bash ./test_tpchc_static.sh" > $log_fn
bash ./test_tpchc_static.sh 2>&1 >> $log_fn
echo ">>>>>>>>>>bash ./test_tpchc_shuffle.sh" >> $log_fn
bash ./test_tpchc_shuffle.sh 2>&1 >> $log_fn
echo ">>>>>>>>>>bash ./test_tpchc_varied.sh" >> $log_fn
bash ./test_tpchc_varied.sh 2>&1 >> $log_fn
echo ">>>>>>>>>>bash ./test_tpchc_varied_shuffle.sh" >> $log_fn
bash ./test_tpchc_varied_shuffle.sh 2>&1 >> $log_fn
echo ">>>>>>>>>>bash ./test_tpchc_params.sh"
bash ./test_tpchc_params.sh 2>&1 >> $log_fn
