#!/bin/bash
    
# python uex_Explain.py --dataset syn1 --input_dim 10 --explainer Grad --model GCN
# python uex_Explain.py --dataset syn1 --input_dim 10 --explainer GNNExplainer --model GCN
# python uex_Explain.py --dataset syn1 --input_dim 10 --explainer Grad --model GIN
# python uex_Explain.py --dataset syn1 --input_dim 10 --explainer GNNExplainer --model GIN

# python uex_Explain.py --dataset syn3 --input_dim 10 --explainer Grad --model GCN
# python uex_Explain.py --dataset syn3 --input_dim 10 --explainer GNNExplainer --model GCN
# python uex_Explain.py --dataset syn3 --input_dim 10 --explainer Grad --model GIN
# python uex_Explain.py --dataset syn3 --input_dim 10 --explainer GNNExplainer --model GIN

# python uex_Explain.py --dataset syn4 --input_dim 10 --explainer Grad --model GCN
# python uex_Explain.py --dataset syn4 --input_dim 10 --explainer GNNExplainer --model GCN
# python uex_Explain.py --dataset syn4 --input_dim 10 --explainer Grad --model GIN
# python uex_Explain.py --dataset syn4 --input_dim 10 --explainer GNNExplainer --model GIN

# python uex_Explain.py --dataset syn5 --input_dim 10 --explainer Grad --model GCN
# python uex_Explain.py --dataset syn5 --input_dim 10 --explainer GNNExplainer --model GCN
# python uex_Explain.py --dataset syn5 --input_dim 10 --explainer Grad --model GIN
# python uex_Explain.py --dataset syn5 --input_dim 10 --explainer GNNExplainer --model GIN

# python uex_Train.py --dataset Cora
# python uex_Train.py --dataset CiteSeer
# python uex_Train.py --dataset PubMed

# python uex_Train.py --dataset Cora --model GAT --lr 0.001
# python uex_Train.py --dataset CiteSeer --model GAT --lr 0.001
# python uex_Train.py --dataset PubMed --model GAT --lr 0.001

# python uex_Explain_citations.py --dataset Cora --explainer GNNExplainer --model GCN
# python uex_Explain_citations.py --dataset PubMed --explainer GNNExplainer --model GCN
# python uex_Explain_citations.py --dataset CiteSeer --explainer GNNExplainer --model GCN

# python uex_Explain_citations.py --dataset Cora --explainer GNNExplainer --model GIN
# python uex_Explain_citations.py --dataset PubMed --explainer GNNExplainer --model GIN
# python uex_Explain_citations.py --dataset CiteSeer --explainer GNNExplainer --model GIN

# python uex_Explain_citations.py --dataset Cora --explainer GNNExplainer --model GAT
# python uex_Explain_citations.py --dataset PubMed --explainer GNNExplainer --model GAT
# python uex_Explain_citations.py --dataset CiteSeer --explainer GNNExplainer --model GAT

python uex_Train.py --dataset syn1 --input_dim 10 --model GAT

python uex_Train.py --dataset syn3 --input_dim 10 --model GAT

python uex_Train.py --dataset syn4 --input_dim 10 --model GAT

python uex_Train.py --dataset syn5 --input_dim 10 --model GAT

python uex_Train.py --dataset syn_prop_1 --input_dim 10 --model GAT

python uex_Train.py --dataset syn_prop_2 --input_dim 10 --model GAT

python uex_Train.py --dataset syn_agg_1 --input_dim 20 --model GAT

python uex_Train.py --dataset syn_agg_2 --input_dim 20 --model GAT

# python uex_Explain.py --dataset syn1 --input_dim 10 --explainer Grad --model GCN
# python uex_Explain.py --dataset syn1 --input_dim 10 --explainer GNNExplainer --model GCN
# python uex_Explain.py --dataset syn1 --input_dim 10 --explainer Grad --model GIN
# python uex_Explain.py --dataset syn1 --input_dim 10 --explainer GNNExplainer --model GIN

# python uex_Explain.py --dataset syn3 --input_dim 10 --explainer Grad --model GCN
# python uex_Explain.py --dataset syn3 --input_dim 10 --explainer GNNExplainer --model GCN
# python uex_Explain.py --dataset syn3 --input_dim 10 --explainer Grad --model GIN
# python uex_Explain.py --dataset syn3 --input_dim 10 --explainer GNNExplainer --model GIN

# python uex_Explain.py --dataset syn4 --input_dim 10 --explainer Grad --model GCN
# python uex_Explain.py --dataset syn4 --input_dim 10 --explainer GNNExplainer --model GCN
# python uex_Explain.py --dataset syn4 --input_dim 10 --explainer Grad --model GIN
# python uex_Explain.py --dataset syn4 --input_dim 10 --explainer GNNExplainer --model GIN

# python uex_Explain.py --dataset syn5 --input_dim 10 --explainer Grad --model GCN
# python uex_Explain.py --dataset syn5 --input_dim 10 --explainer GNNExplainer --model GCN
# python uex_Explain.py --dataset syn5 --input_dim 10 --explainer Grad --model GIN
# python uex_Explain.py --dataset syn5 --input_dim 10 --explainer GNNExplainer --model GIN