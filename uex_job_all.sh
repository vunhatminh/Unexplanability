#!/bin/bash

python uex_GenData.py --dataset syn1 --input_dim 10
python uex_GenData.py --dataset syn3 --input_dim 10
python uex_GenData.py --dataset syn4 --input_dim 10
python uex_GenData.py --dataset syn5 --input_dim 10

python uex_Train.py --dataset syn1 --input_dim 10 --model GCN
python uex_Train.py --dataset syn1 --input_dim 10 --model GIN

python uex_Train.py --dataset syn3 --input_dim 10 --model GCN
python uex_Train.py --dataset syn3 --input_dim 10 --model GIN

python uex_Train.py --dataset syn4 --input_dim 10 --model GCN
python uex_Train.py --dataset syn4 --input_dim 10 --model GIN

python uex_Train.py --dataset syn5 --input_dim 10 --model GCN
python uex_Train.py --dataset syn5 --input_dim 10 --model GIN

python uex_Explain.py --dataset syn1 --input_dim 10 --explainer Grad --model GCN
python uex_Explain.py --dataset syn1 --input_dim 10 --explainer GNNExplainer --model GCN
python uex_Explain.py --dataset syn1 --input_dim 10 --explainer Grad --model GIN
python uex_Explain.py --dataset syn1 --input_dim 10 --explainer GNNExplainer --model GIN

python uex_Explain.py --dataset syn3 --input_dim 10 --explainer Grad --model GCN
python uex_Explain.py --dataset syn3 --input_dim 10 --explainer GNNExplainer --model GCN
python uex_Explain.py --dataset syn3 --input_dim 10 --explainer Grad --model GIN
python uex_Explain.py --dataset syn3 --input_dim 10 --explainer GNNExplainer --model GIN

python uex_Explain.py --dataset syn4 --input_dim 10 --explainer Grad --model GCN
python uex_Explain.py --dataset syn4 --input_dim 10 --explainer GNNExplainer --model GCN
python uex_Explain.py --dataset syn4 --input_dim 10 --explainer Grad --model GIN
python uex_Explain.py --dataset syn4 --input_dim 10 --explainer GNNExplainer --model GIN

python uex_Explain.py --dataset syn5 --input_dim 10 --explainer Grad --model GCN
python uex_Explain.py --dataset syn5 --input_dim 10 --explainer GNNExplainer --model GCN
python uex_Explain.py --dataset syn5 --input_dim 10 --explainer Grad --model GIN
python uex_Explain.py --dataset syn5 --input_dim 10 --explainer GNNExplainer --model GIN

python uex_Train.py --dataset Cora
python uex_Train.py --dataset CiteSeer
python uex_Train.py --dataset PubMed

python uex_Train.py --dataset Cora --model GAT --lr 0.001
python uex_Train.py --dataset CiteSeer --model GAT --lr 0.001
python uex_Train.py --dataset PubMed --model GAT --lr 0.001

python uex_Train.py --dataset Cora --model GIN --lr 0.001
python uex_Train.py --dataset CiteSeer --model GIN --lr 0.001
python uex_Train.py --dataset PubMed --model GIN --lr 0.001

python uex_Explain_citations.py --dataset Cora --explainer GNNExplainer --model GCN
python uex_Explain_citations.py --dataset PubMed --explainer GNNExplainer --model GCN
python uex_Explain_citations.py --dataset CiteSeer --explainer GNNExplainer --model GCN

python uex_Explain_citations.py --dataset Cora --explainer GNNExplainer --model GIN
python uex_Explain_citations.py --dataset PubMed --explainer GNNExplainer --model GIN
python uex_Explain_citations.py --dataset CiteSeer --explainer GNNExplainer --model GIN

python uex_Explain_citations.py --dataset Cora --explainer GNNExplainer --model GAT
python uex_Explain_citations.py --dataset PubMed --explainer GNNExplainer --model GAT
python uex_Explain_citations.py --dataset CiteSeer --explainer GNNExplainer --model GAT


for prob in '0.01' '0.02' '0.03' '0.04' '0.05' '0.06' '0.07' '0.08' '0.09' '0.1'
do
    
    python uex_GenData.py --dataset syn_prop_1 --input_dim 10 --perturb_prob $prob
    python uex_Train.py --dataset syn_prop_1 --input_dim 10 --perturb_prob $prob --model GCN
    python uex_Train.py --dataset syn_prop_1 --input_dim 10 --perturb_prob $prob --model GIN
    
    python uex_GenData.py --dataset syn_prop_2 --input_dim 10 --perturb_prob $prob
    python uex_Train.py --dataset syn_prop_2 --input_dim 10 --perturb_prob $prob --model GCN
    python uex_Train.py --dataset syn_prop_2 --input_dim 10 --perturb_prob $prob --model GIN

    python uex_GenData.py --dataset syn_agg_1 --input_dim 20 --perturb_prob $prob
    python uex_Train.py --dataset syn_agg_1 --input_dim 20 --perturb_prob $prob --model GCN
    python uex_Train.py --dataset syn_agg_1 --input_dim 20 --perturb_prob $prob --model GIN
    
    python uex_GenData.py --dataset syn_agg_2 --input_dim 20 --perturb_prob $prob
    python uex_Train.py --dataset syn_agg_2 --input_dim 20 --perturb_prob $prob --model GCN
    python uex_Train.py --dataset syn_agg_2 --input_dim 20 --perturb_prob $prob --model GIN

done


for prob in '0.01' '0.02' '0.03' '0.04' '0.05' '0.06' '0.07' '0.08' '0.09' '0.1'
do
    python uex_Explain.py --dataset syn_prop_1 --input_dim 10 --perturb_prob $prob --explainer Grad
    python uex_Explain.py --dataset syn_prop_1 --input_dim 10 --perturb_prob $prob
    python uex_Explain.py --dataset syn_prop_2 --input_dim 10 --perturb_prob $prob --explainer Grad
    python uex_Explain.py --dataset syn_prop_2 --input_dim 10 --perturb_prob $prob
    python uex_Explain.py --dataset syn_agg_1 --input_dim 20 --perturb_prob $prob --explainer Grad
    python uex_Explain.py --dataset syn_agg_1 --input_dim 20 --perturb_prob $prob
    python uex_Explain.py --dataset syn_agg_2 --input_dim 20 --perturb_prob $prob --explainer Grad
    python uex_Explain.py --dataset syn_agg_2 --input_dim 20 --perturb_prob $prob
done

for prob in '0.01' '0.02' '0.03' '0.04' '0.05' '0.06' '0.07' '0.08' '0.09' '0.1'
do
    python uex_Explain.py --dataset syn_prop_1 --input_dim 10 --perturb_prob $prob --explainer Grad --model GIN
    python uex_Explain.py --dataset syn_prop_1 --input_dim 10 --perturb_prob $prob --model GIN
    python uex_Explain.py --dataset syn_prop_2 --input_dim 10 --perturb_prob $prob --explainer Grad --model GIN
    python uex_Explain.py --dataset syn_prop_2 --input_dim 10 --perturb_prob $prob --model GIN
    python uex_Explain.py --dataset syn_agg_1 --input_dim 20 --perturb_prob $prob --explainer Grad --model GIN
    python uex_Explain.py --dataset syn_agg_1 --input_dim 20 --perturb_prob $prob --model GIN
    python uex_Explain.py --dataset syn_agg_2 --input_dim 20 --perturb_prob $prob --explainer Grad --model GIN
    python uex_Explain.py --dataset syn_agg_2 --input_dim 20 --perturb_prob $prob --model GIN
done
