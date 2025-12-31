#!bin/bash

lr=(0.00005 0.00025)
nsteps=(128 256 512)
entcoef=(0 0.1 0.2)
cliprange=(0.2 0.3)
gamma=(0.5 0.99)

# conf_file="${1:-experiments/tpchc_index_count.json}"
# conf_file="${1:-experiments/tpcdsc_index_count.json}"
conf_file="${1:-experiments/ceb_conf/ceb_index_count.json}"
case "$conf_file" in
  *"tpch"*)
    idxc=3
    echo "recommend $idxc indexes for tpch bm"
    ;;
  *"tpcds"*)
    idxc=5
    echo "recommend $idxc indexes for tpcds bm"
    ;;
  *"ceb"*)
    idxc=3
    echo "recommend $idxc indexes for ceb bm"
    ;;
esac

for l in "${lr[@]}"; do
  for ns in "${nsteps[@]}"; do
    for  ec in "${entcoef[@]}"; do
      for cr in "${cliprange[@]}"; do
        for ga in "${gamma[@]}"; do
          echo "Running with lr=$l nsteps=$ns entcoef=$ec cliprange=$cr gamma=$ga"
          log_fn=gridsearch_lr${l}_nsteps_${ns}_entcoef_${ec}_cliprange${cr}_gamma${ga}.log
          # echo "python main.py --config $conf_file --fix_index_count $idxc --ts 100000 --tb_log tensor_log/gridsearch --lr $lr --ns $ns --ec $ec --cr $cr --gamma $ga > logs/$log_fn"
          python main.py --config $conf_file --fix_index_count $idxc --ts 100000 --tb_log tensor_log/gridsearch --lr $lr --ns $ns --ec $ec --cr $cr --gamma $ga --num_parallel_env 8 > logs/$log_fn
        done
      done
    done
  done
done
