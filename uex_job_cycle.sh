#!/bin/bash


# Done 0.1 0.

# for prob in '0.001' '0.005' '0.01' 0.02' '0.03' '0.04' '0.06' '0.07' '0.08' '0.09' '0.1'
for prob in '0.001' '0.005' '0.02' '0.03' '0.04' '0.05' '0.06' '0.07' '0.08' '0.09' '0.1'
do
    
    python uex_GenData.py --dataset syn_prop_1 --input_dim 10 --perturb_prob $prob
    python uex_Train.py --dataset syn_prop_1 --input_dim 10 --perturb_prob $prob --model GCN
    python uex_Train.py --dataset syn_prop_1 --input_dim 10 --perturb_prob $prob --model GIN
#     python uex_Explain.py --dataset syn_prop_1 --input_dim 10 --perturb_prob $prob --model GCN
#     python uex_Explain.py --dataset syn_prop_1 --input_dim 10 --perturb_prob $prob --model GIN

    python uex_GenData.py --dataset syn_agg_1 --input_dim 20 --perturb_prob $prob
    python uex_Train.py --dataset syn_agg_1 --input_dim 20 --perturb_prob $prob --model GCN
    python uex_Train.py --dataset syn_agg_1 --input_dim 20 --perturb_prob $prob --model GIN
#     python uex_Explain.py --dataset syn_agg_1 --input_dim 20 --perturb_prob $prob --model GCN
#     python uex_Explain.py --dataset syn_agg_1 --input_dim 20 --perturb_prob $prob --model GIN

done