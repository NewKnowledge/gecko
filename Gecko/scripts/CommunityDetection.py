from Gecko import Gecko
from gem.utils import graph_util

import networkx as nx

# GRAPH RECONSTRUCTION
# File that contains the edges. Format: source target
# edge_f = 'data/karate.edgelist'

# *.gml inputs can be processed as follows
# graphpath = "data/6_70_com_amazon.gml"
# graph = nx.read_gml(graphpath)
edge_f = 'data/6_70_com_amazon.edgelist'
# nx.write_edgelist(graph,edge_f,data=False)

# edge_f = 'data/BUP_train_0.net'

# Specify whether the edges are directed
isDirected = False # crucial for performance
# Load graph
G_original = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=isDirected)
G = G_original.to_directed()
# important that nodes are contiguously numbered
G = nx.convert_node_labels_to_integers(G,first_label=0,ordering='default',label_attribute="original_label")

embedding_generator = Gecko()
bestEmbedding = embedding_generator.GraphReconstruction(G=G,visualize=False)
print("DEBUG::The best embedding found is")
print(bestEmbedding)
# advance users can also define their own embeddings directly, as done in the __init__ method above

# Community Detection/ Node Clustering using Graph Embeddings
communities = embedding_generator.CommunityDetection(G=G,embedding=bestEmbedding,visualize=True)
print("DEBUG::The labels from GE-based community detection are:")
print(communities.labels_)

# Community Detection/ Node Clustering using Louvain alternative
communities = embedding_generator.CommunityDetectionLouvain(G=G_original,visualize=True)
print("DEBUG::The labels from Louvain community detection are:")
print(list(communities.values()))