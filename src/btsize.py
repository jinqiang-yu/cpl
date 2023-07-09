#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import print_function
from data import Data
from options import Options
import os
import sys
from xgbooster import XGBooster, preprocess_dataset, discretize_dataset, tree
import pandas as pd
import numpy as np
import six
import glob
import collections
import pickle
from options import Options

def traverse(tree, nof_node=0, nof_leaf=0):
    nof_node += 1
    if tree.children:
        nof_node, nof_leaf = traverse(tree.children[0], nof_node, nof_leaf)
        nof_node, nof_leaf = traverse(tree.children[1], nof_node, nof_leaf)
    else:
        nof_leaf += 1

    return nof_node, nof_leaf

if __name__ == '__main__':
    options = Options(sys.argv)
    options.encode = 'maxsat'

    xgb = XGBooster(options, from_model=options.files[0])

    # encode it and save the encoding to another 
    xgb.encode(test_on=options.explain)

    ensemble = tree.TreeEnsemble(xgb.model,
                xgb.extended_feature_names_as_array_strings,
                nb_classes=xgb.num_class)
    
    nof_nodes = []
    nof_leafs = []
    # traversing and encoding each tree
    for i, tree in enumerate(ensemble.trees):
        nof_node, nof_leaf = traverse(tree)
        nof_nodes.append(nof_node)
        nof_leafs.append(nof_leaf)
    
    print(nof_nodes)
    print(len(nof_nodes))
    print(nof_leafs)

    exit()


