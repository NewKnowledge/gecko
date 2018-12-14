import matplotlib
matplotlib.use('Agg')

import os
os.environ["THEANO_FLAGS"]="mode=FAST_RUN,device=gpu,floatX=float32"

import matplotlib.pyplot as plt

from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr
from time import time

from gem.embedding.gf import GraphFactorization
from gem.embedding.hope import HOPE
from gem.embedding.lap import LaplacianEigenmaps
from gem.embedding.lle import LocallyLinearEmbedding

import networkx as nx

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# File that contains the edges. Format: source target
# Optionally, you can add weights as third column: source target weight, but we are not using this for now
graphpath = "data/6_70_com_amazon.gml"
graph = nx.read_gml(graphpath)
edge_f = 'data/6_70_com_amazon.edgelist'
nx.write_edgelist(graph,edge_f,data=False)

edge_f = 'data/karate.edgelist'


# Specify whether the edges are directed
isDirected = False

# Load graph
G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=isDirected)
G = G.to_directed()

G = nx.convert_node_labels_to_integers(G,first_label=0,ordering='default',label_attribute="original_label")

models = []
# models.append(GraphFactorization(d=2, max_iter=100000, eta=1*10**-4, regu=1.0))
models.append(HOPE(d=4, beta=0.01))
# models.append(LaplacianEigenmaps(d=2))
# models.append(LocallyLinearEmbedding(d=2))

for embedding in models:
    print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
    t1 = time()
    # Learn embedding - accepts a networkx graph or file with edge list
    Y, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
    print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
    # Evaluate on graph reconstruction
    MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)
    #---------------------------------------------------------------------------------
    print(("\tMAP: {} \t precision curve: {}\n\n\n\n"+'-'*100).format(MAP,prec_curv[:5]))
    #---------------------------------------------------------------------------------
    
    print("DEBUG::embedding vectors:")
    print(embedding.get_embedding())

    # cluster vectors (community detection!)
    kmeans = KMeans(n_clusters=2,random_state=0).fit(embedding.get_embedding())
    print(kmeans.labels_)

    # Visualize using GEM library
    X_=embedding.get_embedding()
    viz.plot_embedding2D(X_, di_graph=G, node_colors=None)
    plt.show()

    # Visualize using tsne directly
    X_= TSNE(n_components=2).fit_transform(X_)
    print("DEBUG::embedding vectors:")
    print(X_)
    pos ={}
    n_nodes = X_.shape[0]
    for i in range(n_nodes):
        pos[i] = X_[i,:]
    nx.draw_networkx(G,pos,node_color=kmeans.labels_,node_size=300,alpha=0.5,arrows=False,font_size=12)
    plt.title('Community Detection using GEMs on Zacharys Karate Club Graph')
    plt.show()