# /workspace/src/planning/batch_run.sh
#!/bin/bash

for iters in 1 2 5 20 50 100 150
do
    echo "Running outer_iters=$iters"
    python /home/qiandaoliu/workspace/src/planning/run_mpc_physenv.py \
        --init_idx 0 --target_idx 0 --outer_iters $iters
done
