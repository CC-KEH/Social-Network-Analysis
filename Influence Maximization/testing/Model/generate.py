import os
import networkx as nx
import pandas as pd
import numpy as np
import glob

from graph_generator import graph_generator

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Create the path to the directory where you want to save the graphs
graphs_dir = os.path.join(script_dir, "datasets", "sim_graphs")

# Create the directory if it doesn't exist
if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)

generator = graph_generator()
generator.gen_new_graphs(300,500,30)    
generator.save_graphs(graphs_dir)  # Save graphs in the created directory

os.chdir(graphs_dir)  # Change working directory to the created directory

# make them undirected
for g in glob.glob("*"):#
    G = pd.read_csv(g,header=None,sep=" ")
    G.columns = ["node1","node2","w"]
    del G["w"]
    # make undirected directed
    tmp = G.copy()
    G = pd.DataFrame(np.concatenate([G.values, tmp[["node2","node1"]].values]),columns=G.columns)
    
    G.columns = ["source","target"]
    
    outdegree = G.groupby("target").agg('count').reset_index()
    outdegree.columns = ["target","weight"]
    
    outdegree["weight"] = 1/outdegree["weight"]
    outdegree["weight"] = outdegree["weight"].apply(lambda x:float('%s' % float('%.6f' % x)))
    G = G.merge(outdegree, on="target")
    G.to_csv(g,sep=" ",header=None,index=False)

