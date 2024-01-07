""" featgen.py

Node feature generators.

"""
import networkx as nx
import numpy as np
import random

import abc


class FeatureGen(metaclass=abc.ABCMeta):
    """Feature Generator base class."""
    @abc.abstractmethod
    def gen_node_features(self, G):
        pass


class ConstFeatureGen(FeatureGen):
    """Constant Feature class."""
    def __init__(self, val):
        self.val = val

    def gen_node_features(self, G):
        feat_dict = {i:{'x': np.array(self.val, dtype=np.float32)} for i in G.nodes()}
#         print ('feat_dict[0]["feat"]:', feat_dict[0]['feat'].dtype)
        nx.set_node_attributes(G, feat_dict)
#         print ('G.nodes[0]["feat"]:', G.nodes[0]['feat'].dtype)


class GaussianFeatureGen(FeatureGen):
    """Gaussian Feature class."""
    def __init__(self, mu, sigma):
        self.mu = mu
        if sigma.ndim < 2:
            self.sigma = np.diag(sigma)
        else:
            self.sigma = sigma

    def gen_node_features(self, G):
        feat = np.random.multivariate_normal(self.mu, self.sigma, G.number_of_nodes())
        # Normalize feature
        feat = (feat+np.max(np.abs(feat)))/np.max(np.abs(feat))/2
        feat_dict = {
                i: {"x": feat[i]} for i in range(feat.shape[0])
            }
        nx.set_node_attributes(G, feat_dict)
        
class ConstFeatureWithRoleGen(FeatureGen):
    def __init__(self, feat_vals, input_dim = 10):
        self.feat_vals = feat_vals
        self.input_dim = input_dim

    def gen_node_features(self, G):
        
        feat_dict = {i:{'x': np.array([self.feat_vals[i] for _ in range(self.input_dim)], dtype=np.float32)} for i in G.nodes()}
        nx.set_node_attributes(G, feat_dict)

class SynP1RoleGen(FeatureGen):
    def __init__(self, feat_vals, input_dim = 10):
        self.feat_vals = feat_vals
        self.input_dim = input_dim
        self.role_encode_dim = int(np.max(feat_vals)) + 1
        self.random_dim = input_dim - self.role_encode_dim

    def gen_node_features(self, G):
        one_hot = np.zeros((self.feat_vals.size, self.input_dim))
        one_hot[np.arange(self.feat_vals.size),self.feat_vals] = 1 
        feat_dict = {i:{'x': one_hot[i].astype(float)} for i in G.nodes()}                   
        nx.set_node_attributes(G, feat_dict)
        
class SynA1RoleGen(FeatureGen):
    def __init__(self, feat_vals, input_dim = 10):
        self.feat_vals = feat_vals
        self.input_dim = input_dim
        self.role_encode_dim = int(np.max(feat_vals)) + 1
        self.random_dim = input_dim - self.role_encode_dim

    def gen_node_features(self, G):       
        one_hot = np.random.randint(2, size =(self.feat_vals.size, self.input_dim))
        one_hot[np.arange(self.feat_vals.size),:self.role_encode_dim] = 0
        one_hot[np.arange(self.feat_vals.size),self.feat_vals] = 1 
        feat_dict = {i:{'x': one_hot[i].astype(float)} for i in G.nodes()}                   
        nx.set_node_attributes(G, feat_dict)
        
class RandomFeatureWithRoleGen(FeatureGen):
    """Gaussian Feature class."""
    def __init__(self, feat_vals, input_dim = 10):
        self.feat_vals = feat_vals
        self.input_dim_role = int(2)
        self.input_dim_val = int(input_dim - self.input_dim_role)
    
    def gen_node_features(self, G):
        feat_dict = {}
        for i in G.nodes():
            role = [self.feat_vals[i] for _ in range(self.input_dim_role)] 
            val = [np.random.randint(0,2,1) for _ in range(self.input_dim_val)] 
            feat_dict[i] = {'x': np.asarray(role + val,  dtype=np.float32)}

        nx.set_node_attributes(G, feat_dict)
        
class RandomFeatureGen(FeatureGen):
    """Gaussian Feature class."""
    def __init__(self, input_dim = 10):
        self.input_dim = input_dim
    
    def gen_node_features(self, G):
        feat_dict = {}
        for i in G.nodes():
            feat_dict[i] = {'x': np.random.randint(0,2,self.input_dim).astype(np.float32)}
        nx.set_node_attributes(G, feat_dict)


class GridFeatureGen(FeatureGen):
    """Grid Feature class."""
    def __init__(self, mu, sigma, com_choices):
        self.mu = mu                    # Mean
        self.sigma = sigma              # Variance
        self.com_choices = com_choices  # List of possible community labels

    def gen_node_features(self, G):
        # Generate community assignment
        community_dict = {
            n: self.com_choices[0] if G.degree(n) < 4 else self.com_choices[1]
            for n in G.nodes()
        }

        # Generate random variable
        s = np.random.normal(self.mu, self.sigma, G.number_of_nodes())

        # Generate features
        feat_dict = {
            n: {"x": np.asarray([community_dict[n], s[i]])}
            for i, n in enumerate(G.nodes())
        }

        nx.set_node_attributes(G, feat_dict)
        return community_dict
