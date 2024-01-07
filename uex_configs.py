import argparse
import parser_utils
    
def arg_parse():
    parser = argparse.ArgumentParser(description='Unexplanable proof arguments.')

    parser.add_argument('--dataset', dest='dataset',
            help='The experiment')
    parser.add_argument('--perturb_prob', dest='perturb_prob', type=float,
            help='Probability that the graph-data-edges are perturbed.')
    parser.add_argument('--model', dest='model',
            help='The model type')
    parser.add_argument('--explainer', dest='explainer',
            help='The explainer')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--train_ratio', dest='train_ratio', type=float,
            help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--input_dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
            help='Weight decay regularization constant.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')

    parser.set_defaults(
#                         datadir='data', # io_parser
#                         logdir='log',
#                         ckptdir='ckpt',
                        dataset='syn_prop_1',
                        perturb_prob=0.01,
                        model='GCN',
                        explainer='GNNExplainer',
#                         opt_scheduler='none',
#                         max_nodes=100,
#                         cuda='1',
#                         feature_type='default',
                        lr=0.001,
#                         clip=2.0,
#                         batch_size=20,
                        num_epochs=1000,
                        train_ratio=0.8,
#                         test_ratio=0.1,
#                         num_workers=1,
                        input_dim=10,
#                         hidden_dim=20,
#                         output_dim=20,
#                         num_classes=2,
#                         num_gc_layers=3,
                        dropout=0.0,
                        weight_decay=0.005,
#                         method='base',
#                         name_suffix='',
#                         assign_ratio=0.1,
                       )
    return parser

