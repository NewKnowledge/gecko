from Gecko import Gecko
from gem.utils import graph_util

import networkx as nx

# GRAPH RECONSTRUCTION
# File that contains the edges. Format: source target
# edge_f = 'data/karate.edgelist'
# edge_f = 'data/BUP_train_0.net'

# *.gml inputs can be processed as follows
graphpath = "data/LL1_bn_fly_drosophila_medulla_net.gml"
graph = nx.read_gml(graphpath)
edge_f = 'data/LL1_bn_fly_drosophila_medulla_net.edgelist'
nx.write_edgelist(graph,edge_f,data=False)


# Specify whether the edges are directed
isDirected = False # crucial for performance
# Load graph
G = graph_util.loadGraphFromEdgeListTxt(edge_f,directed=isDirected)

embedding_generator = Gecko()
bestEmbedding = embedding_generator.GraphReconstruction(G=G,visualize=False,directed=isDirected)
print("DEBUG::The best embedding found is")
print(bestEmbedding)
# advance users can also define their own embeddings directly, by passing embedding to Gecko constructor

# Community Detection/ Node Clustering using Graph Embeddings
communities = embedding_generator.CommunityDetection(G=G,embedding=bestEmbedding,n_clusters=4,visualize=True)
print("DEBUG::The labels from GE-based community detection are:")
print(communities.labels_)

# Community Detection/ Node Clustering using Louvain alternative -- input **must** be undirected
communities = embedding_generator.CommunityDetectionLouvain(G=G,visualize=True)
print("DEBUG::The labels from Louvain community detection are:")
print(list(communities.values()))