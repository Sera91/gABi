import numpy as np
import pandas as pd
import pickle
import sys
import os
import gc
import time
import networkx as nx
import matplotlib.pyplot as plt
from itertools import chain, combinations, permutations
from coreBN.utils import GAM_residuals, GAM_residuals_fast
#import coreBN
from coreBN.CItests import kernel_CItest_cycle
#from coreBN.estimators.PC import kPC as kPC
from coreBN.base import PDAG
import params_basic_1000 as params



# %% Make plot
def plot(G, node_color=None, node_size=1500, node_size_scale=[80, 1000], alpha=0.8, font_size=16, cmap='Set2', width=30, height=30, pos=None, filename=None, title=None, methodtype='circular', layout='circular_layout', verbose=3):
    # https://networkx.github.io/documentation/networkx-1.7/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html
    config = {}
    config['filename']=filename
    config['width']=width
    config['height']=height
    config['verbose']=verbose
    config['node_size_scale']=node_size_scale

    if verbose>=3: print('[gABiC] >Creating network plot')

    ##### DEPRECATED IN LATER VERSION #####
    if methodtype is not None:
        if verbose>=2: print('[gABiC] >Methodtype will be removed in future version. Please use "layout" instead')
        if methodtype=='circular':
            layout = 'draw_circular'
        elif methodtype=='kawai':
            layout = 'draw_kamada_kawai'
        else:
            layout = 'spring_layout'
    ##### END BLOCK #####

    if 'pandas' in str(type(node_size)):
        node_size=node_size.values

    # scaling node sizes
    if config['node_size_scale']!=None and 'numpy' in str(type(node_size)):
        if verbose>=3: print('[gABiC] >Scaling node sizes')
        node_size=minmax_scale(node_size, feature_range=(node_size_scale[0], node_size_scale[1]))

    # Setup figure
    fig = plt.figure(figsize=(config['width'], config['height']))

    # Make the graph
    try:
        # Get the layout
        layout_func = getattr(nx, layout)
        layout_func(G, labels=node_label, node_size=1000, alhpa=alpha, node_color=node_color, cmap=cmap, font_size=font_size, with_labels=True)
    except:
        if verbose>=2: print('[gABiC] >Warning: [%s] layout not found. The [spring_layout] is used instead.' %(layout))
        #nx.spring_layout(G, labels=node_label, pos=pos, node_size=node_size, alhpa=alpha, node_color=node_color, cmap=cmap, font_size=font_size, with_labels=True)
    if methodtype=='spring':
       nx.draw(G, pos=nx.spring_layout(G), node_size=node_size, alpha=alpha, node_color='white', edgecolors="black", font_size=font_size, with_labels=True)

    if methodtype=='circular':
       nx.draw(G, pos=nx.circular_layout(G), node_size=node_size, alpha=alpha, node_color=node_color, cmap=cmap, font_size=font_size, with_labels=True)
    elif methodtype=='kawai':
       nx.draw(G, pos=nx.kamada_kawai_layout(G), node_size=node_size, alpha=alpha, node_color='white', edgecolors="black", font_size=font_size, with_labels=True)
    #     nx.draw_kamada_kawai(G, labels=node_label, node_size=node_size, alhpa=alpha, node_color=node_color, cmap=cmap, font_size=font_size, with_labels=True)
    # else:
        # nx.draw_networkx(G, labels=node_label, pos=pos, node_size=node_size, alhpa=alpha, node_color=node_color, cmap=cmap, font_size=font_size, with_labels=True)

    plt.title(title)
    plt.grid(True)
    plt.show()

    # Savefig
    if not isinstance(config['filename'], type(None)):
        if verbose>=3: print('[gABiC] >Saving figure')
        plt.savefig(config['filename'])

    return(fig)


edges= [('x1', 'x3'), ('x1', 'x2'), ('x9', 'x7'), ('x7', 'x8'), ('x3', 'x5'),('x4', 'x5'), ('x6', 'x7')]

G = nx.DiGraph()

G.add_edges_from(edges)


plt.figure(1, figsize=(25,25))
pos_kama=nx.kamada_kawai_layout(G)
pos_viz = nx.nx_agraph.graphviz_layout(G, prog="dot")
nx.draw(G, pos=pos_viz , node_size=1000, alpha=0.7, node_color='white', edgecolors="black", font_size=16, with_labels=True)
plt.savefig("base_DAG.png")
plt.show()




