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
def plot(G, node_color=None, node_size=1500, node_size_scale=[80, 1000], alpha=0.8, font_size=16, cmap='Set2', width=25, height=25, pos=None, filename=None, title=None, methodtype='circular', layout='circular_layout', verbose=3):
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
    if methodtype='graphviz':
       pos_viz = nx.nx_agraph.graphviz_layout(G, prog="dot")
       nx.draw(G, pos=pos_viz , node_size=1000, alpha=0.7, node_color='white', edgecolors="black", font_size=16, with_labels=True)
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



def build_skeleton(
        data,
        variables,
        ci_test,
        rstate,
        significance_level,
        verbose=False
    ):
        """
        Estimates a graph skeleton (UndirectedGraph) from a set of independencies
        using (the first part of) the PC algorithm. The independencies can either be
        provided as an instance of the `Independencies`-class or by passing a
        decision function that decides any conditional independency assertion.
        Returns a tuple `(skeleton, separating_sets)`.

        If an Independencies-instance is passed, the contained IndependenceAssertions
        have to admit a faithful BN representation. This is the case if
        they are obtained as a set of d-seperations of some Bayesian network or
        if the independence assertions are closed under the semi-graphoid axioms.
        Otherwise the procedure may fail to identify the correct structure.

        Parameters
        ----------

        Returns
        -------
        skeleton: UndirectedGraph
            An estimate for the undirected graph skeleton of the BN underlying the data.

        separating_sets: dict
            A dict containing for each pair of not directly connected nodes a
            separating set ("witnessing set") of variables that makes then
            conditionally independent. (needed for edge orientation procedures)

        References
        ----------
        [1] Neapolitan, Learning Bayesian Networks, Section 10.1.2, Algorithm 10.2 (page 550)
            http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
        [2] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
            Section 3.4.2.1 (page 85), Algorithm 3.3

        Examples
        --------
        >>> from coreBN.estimators import PC
        >>> from coreBN.base import DAG
        >>> from coreBN.independencies import Independencies
        >>> # build skeleton from list of independencies:
        ... ind = Independencies(['B', 'C'], ['A', ['B', 'C'], 'D'])
        >>> # we need to compute closure, otherwise this set of independencies doesn't
        ... # admit a faithful representation:
        ... ind = ind.closure()
        >>> skel, sep_sets = PC(independencies=ind).build_skeleton("ABCD", ind)
        >>> print(skel.edges())
        [('A', 'D'), ('B', 'D'), ('C', 'D')]
        >>> # build skeleton from d-seperations of DAG:
        ... model = DAG([('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'E')])
        >>> skel, sep_sets = PC.build_skeleton(model.nodes(), model.get_independencies())
        >>> print(skel.edges())
        [('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'E')]
        """

        # Initialize initial values and structures.
        #lim_neighbors = 0
        #separating_sets = dict()
        if ci_test == "hsic_gamma":
            sel_method="hsic.gamma"
        elif ci_test == "dcc_gamma":
            sel_method="dcc.gamma"
        else:
            raise ValueError(
                f"ci_test must be either hsic_gamma, dcc_gamma or a function. Got: {ci_test}")
        
        max_cond_vars= len(variables)-2
        
        if verbose:        
            print("I have created the moral graph")
        # Step 1: Initialize a fully connected undirected graph
        graph = nx.complete_graph(n=variables, create_using=nx.Graph)
        if verbose:
            print("graph edges:")
            print(list(graph.edges()))
        
        
        separating_sets = dict() #dictionary where we will save each separating set for pair of nodes


        # Exit condition: 1. If all the nodes in graph has less than `lim_neighbors` neighbors.
        #             or  2. `lim_neighbors` is greater than `max_conditional_variables`.	


        #saved_pk_file = "sepsets-TNG300-seldisks-at-step2_rstate5_4n.pkl"
        #file_to_read = open(saved_pk_file, "rb")
        #separating_sets = pickle.load(file_to_read)
        #indep_edges = list([tuple(elem) for elem in separating_sets.keys()])
        #for edge in indep_edges:
        #    u,v= edge
        #    graph.remove_edge(u,v)



        lim_neighbors = 0
        
        while not all([len(list(graph.neighbors(var))) < lim_neighbors for var in variables]) and lim_neighbors <= max_cond_vars:

                # Step 2: Iterate over the edges and find a conditioning set of
                # size `lim_neighbors` which makes u and v independent.
                 
                neighbors = {node: set(graph[node]) for node in graph.nodes()}
                list_edges = list(graph.edges())
                if verbose:
                    print(("edges at step"+str(lim_neighbors)+":"), list_edges)
                list_results=list([])	
                if (lim_neighbors > 0):
                    list_sel_edges=[(u,v) for (u,v) in list_edges if (len(list(neighbors[u]-set([v])))>=lim_neighbors or len(list(neighbors[v]-set([u])))>=lim_neighbors)]
                else:
                    list_sel_edges=list_edges
                
                
                #list_sel_edges = list([(u,v) for (u,v) in list_edges if ((len(list(graph.neighbors(u)))>0) or (len(list(graph.neighbors(v)))>0)) ])
                list_sepsets = list([set(list(chain(combinations( set(neighbors[u]) - set([v]), lim_neighbors), combinations( set(neighbors[v]) - set([u]), lim_neighbors),))) for (u,v) in list_sel_edges])

                for i, edge in enumerate(list_sel_edges):
                    u, v = edge
                    sep_sets_edge = list_sepsets[i] 
                    edge, sep_set, p_value = kernel_CItest_cycle(u,v,sep_sets_edge, lim_neighbors, significance_level, 0, data, method=sel_method )
                    # If a conditioning set exists stores the edge to remove, store the
                    # separating set and move on to finding conditioning set for next edge
                    if ( p_value > significance_level) :
                                        separating_sets[frozenset((u,v))]= sep_set
                                        graph.remove_edge(u,v)
                                               
       
                del list_sepsets
                gc.collect()
                lim_neighbors += 1

        if verbose:       
            print("skeleton has been built!")
        return graph, separating_sets


def skeleton_to_pdag(skeleton, separating_sets):
        """Orients the edges of a graph skeleton based on information from
        `separating_sets` to form a DAG pattern (DAG).

        Parameters
        ----------
        skeleton: UndirectedGraph
            An undirected graph skeleton as e.g. produced by the
            estimate_skeleton method.

        separating_sets: dict
            A dict containing for each pair of not directly connected nodes a
            separating set ("witnessing set") of variables that makes then
            conditionally independent. (needed for edge orientation)

        Returns
        -------
        Model after edge orientation: base.pDAG
            An estimate for the pDAG of the BN underlying the data. The
            graph might contain some nodes with both-way edges (X->Y and Y->X).
            Any completion by (removing one of the both-way edges for each such
            pair) results in a I-equivalent Bayesian network DAG.

        References
        ----------
        Neapolitan, Learning Bayesian Networks, Section 10.1.2, Algorithm 10.2 (page 550)
        http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf


        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from gABI.estimators import PC
        >>> data = pd.DataFrame(np.random.randint(0, 4, size=(5000, 3)), columns=list('ABD'))
        >>> data['C'] = data['A'] - data['B']
        >>> data['D'] += data['A']
        >>> c = PC(data)
        >>> pdag = c.skeleton_to_pdag(*c.build_skeleton())
        >>> pdag.edges() # edges: A->C, B->C, A--D (not directed)
        [('B', 'C'), ('A', 'C'), ('A', 'D'), ('D', 'A')]
        """

        # applying function from Networkx library to bidirect the edges in the graph

        pdag = skeleton.to_directed()
        node_pairs = list(permutations(pdag.nodes(), 2))

        # 1) for each X-Z-Y, if Z not in the separating set of X,Y, then orient edges as X->Z<-Y
        # (Algorithm 3.4 in Koller & Friedman PGM, page 86)
        for pair in node_pairs:
            X, Y = pair
            if not skeleton.has_edge(X, Y):
                for Z in set(skeleton.neighbors(X)):
                    if (Z in set(skeleton.neighbors(Y))) and (Z not in separating_sets[frozenset((X, Y))]):
                        pdag.remove_edges_from([(Z, X), (Z, Y)])

        # TODO: This is temp fix to get a PDAG object.
        edges = pdag.edges()
        undirected_edges = []
        directed_edges = []
        for u, v in edges:
            if (v, u) in edges:
                undirected_edges.append((u, v))
            else:
                directed_edges.append((u, v))
        return PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)

def regrVonPS(data, pdag, ci_test, significance_level=0.06, verbose=False):
        """
        Input Parameters
        -----------------
        - data : input data 
        - pdag : partially directed grap in output of the search for immoralities
                in the 2nd step of the kpc algorithm
        - ci_test: type of conditional independence test (dcc.gamma or hsic.gamma)
        - alpha : significance level, to evaluate pval from CI test
        -----------------
        Returns:


        This function implements the generalized transitive step
        of kpc algorithm in kpcalg (Verbyla, 2017).
        It first builds a residuals matrix,cycling on the nodes
        in the pDAG and using pyGAM_residuals() to regress each V(node)
        on the variables in the neighborhood of V
        PyGAM uses the generalised additive models to non-linearly
        and non-parametrically fit variables dependence, and here 
        we model all the dependencies only with cubic B-splines. 
        The second step of the function is cycling over undirected
        edges, and estimating pval_L = (r_x \indep y | S) and
        pval_R = (r_y \indep x | S). 
        We can have different outcomes:
            - pval_L > alpha .

        """
        if ci_test == "hsic_gamma":
            from coreBN.CItests import Hsic_gamma_py as CItest
        elif ci_test == "dcc_gamma":
            from coreBN.CItests import Dcov_gamma_py_h as CItest
        else:
            raise ValueError(
                f"ci_test must be either hsic_gamma, dcc_gamma, hsic_perm or dcc_perm or a function. Got: {ci_test}")

    
        #data = pd.read_csv(self.data_input)
        if verbose:
            print("selected method for CI test is:", ci_test)
        residuals_matrix = np.zeros(data.shape, dtype=np.float64)
        # I will use a mapping array to convert string into col index
        # for the residuals matrix

        # this can be avoided defining a recarray in the form:
        #residual_recarray= np.zeros(self.data.shape[0], dtype=list(zip(list(graph.nodes), [np.float64]*self.data.shape[1])))

        for i, node in enumerate(pdag.nodes()):
            if verbose:
                print('estimating residuals of var:', node)
                print('nodes in neighborhood[',node,']:', list(pdag.neighbors(node)))
                print('parents [',node, ']', list(pdag.predecessors(node)))
                print('children[',node, ']', list(pdag.successors(node)))
            set_Z = list(pdag.predecessors(node))
            S_Z = len(set_Z)
            if (S_Z!=0):
                y_arr = (data[node]).to_numpy()
                data_Sset = (data[set_Z]).to_numpy()
                residuals_matrix[:, i] = GAM_residuals_fast(data_Sset, y_arr, S_Z)
            else:
                print("this node has no possible predecessors")
        if verbose:
            print("nodes in pdag:", list(pdag.nodes()))
            print("all edges in pdag:",list(pdag.edges()))
            print("undirected edges in pdag", list(pdag.undirected_edges))

        screened_edges=[]
        arr_names_nodes = np.array(list(pdag.nodes()))
        for l,r in pdag.undirected_edges:
            if verbose:
                print("working on edge:", l, ' ->', r)
            screened_edges.append((l,r))
            if (r,l) in screened_edges:
                continue
                
            left_vert = l
            right_vert = r
            index_l = np.argwhere(arr_names_nodes == left_vert)[0][0]
            index_r = np.argwhere(arr_names_nodes == right_vert)[0][0]
            res_left = residuals_matrix[:, index_l]
            res_right = residuals_matrix[:, index_r]
            x_left, y_left = res_left, (data[right_vert]).to_numpy()
            x_right, y_right = res_right, (data[left_vert]).to_numpy()

            #setting default pars for CI tests

            p_left = CItest(x_left, y_left,0)
            p_right = CItest(x_right, y_right, 0)
            
            if ((p_left > significance_level) & (p_right < significance_level)):
                # removes edge from undirected
                # leaves edge right-> left as directed
                pdag.remove_edge(left_vert, right_vert)
            if ((p_right > significance_level) & (p_left < significance_level)):
                # removes edge from undirected
                # leaves edge left-> right as directed
                pdag.remove_edge(right_vert, left_vert)

        edges = set(pdag.edges())
        undirected_edges = []
        directed_edges = []
        for u, v in edges:
            if (v, u) in edges:
                undirected_edges.append((u, v))
            else:
                directed_edges.append((u, v))
        return PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)


def PDAG_by_Meek_rules(pdag):
        """
        this function applies Meek rules to orient edges in the pdag 
        """
        node_pairs = list(permutations(pdag.nodes(), 2))
        # MEEK's RULES
        progress = True
        while progress:  # as long as edges can be oriented ( or removed)
            num_edges = pdag.number_of_edges()

            # 2) for each X->Z-Y, orient edges to Z->Y
            # (Explanation in Koller & Friedman PGM, page 88)
            for pair in node_pairs:
                X, Y = pair
                if not pdag.has_edge(X, Y):
                    for Z in (set(pdag.successors(X)) - set(pdag.predecessors(X))) & (
                        set(pdag.successors(Y)) & set(pdag.predecessors(Y))
                    ):
                        pdag.remove_edge(Y, Z)

            # 3) for each X-Y with a directed path from X to Y, orient edges to X->Y
            for pair in node_pairs:
                X, Y = pair
                if pdag.has_edge(Y, X) and pdag.has_edge(X, Y):
                    for path in nx.all_simple_paths(pdag, X, Y):
                        is_directed = True
                        for src, dst in list(zip(path, path[1:])):
                            if pdag.has_edge(dst, src):
                                is_directed = False
                        if is_directed:
                            pdag.remove_edge(Y, X)
                            break

            # 4) for each X-Z-Y with X->W, Y->W, and Z-W, orient edges to Z->W
            for pair in node_pairs:
                X, Y = pair
                for Z in ( set(pdag.successors(X))
                    & set(pdag.predecessors(X))
                    & set(pdag.successors(Y))
                    & set(pdag.predecessors(Y))
                ):
                    for W in (
                        (set(pdag.successors(X)) - set(pdag.predecessors(X)))
                        & (set(pdag.successors(Y)) - set(pdag.predecessors(Y)))
                        & (set(pdag.successors(Z)) & set(pdag.predecessors(Z)))
                    ):
                        pdag.remove_edge(W, Z)

            progress = num_edges > pdag.number_of_edges()

        # TODO: This is temp fix to get a PDAG object.
        edges = pdag.edges()
        undirected_edges = []
        directed_edges = []
        for u, v in edges:
            if (v, u) in edges:
                undirected_edges.append((u, v))
            else:
                directed_edges.append((u, v))
        return PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)

def estimate(data_TNG, variables, ici_test, random_seed, significance_level=0.05, debug=True):
        if debug:
            print('I am building the graph skeleton')

        t_startP1 = time.perf_counter() 
        # Step 1: Run the PC algorithm to build the skeleton and get the separating sets.	
        skel, separating_sets = build_skeleton(data_TNG, variables, ici_test, rstate=random_seed, significance_level=0.05)
        t_endP1 =  time.perf_counter()
        
        
        if debug:
            print("AFTER STEP 1:")
            print("list nodes in skeleton:", skel.nodes())
            print("list edges in skeleton:", skel.edges())

        t_startP2=time.perf_counter()
        # Step 2: Find V structures/Immoralities	
        pdag = skeleton_to_pdag(skel, separating_sets)
        t_endP2 = time.perf_counter()

        if debug:
            print("PRINT after call to skeleton to PDAG")
            print("list nodes: ", pdag.nodes())
            print("list directed edges: ", pdag.directed_edges)
            print("list undirected edges: ", pdag.undirected_edges )


        # Step 3: Generalized transitive phase
        t_startP3 = time.perf_counter()
        pdag_new = regrVonPS(data_TNG, pdag, ici_test, significance_level=0.06)
        #N_vars= len(df_basic.columns)

        # Step 4: Applying Meek's rules

        complete_pdag = PDAG_by_Meek_rules(pdag_new)
        delta_tP3 = time.perf_counter() -t_startP3
        delta_tP1 = t_endP1- t_startP1
        delta_tP2 = t_endP2- t_startP2 
        print("time build skeleton (s) :  {:12.6f}".format(delta_tP1))
        print("time phase 2 (s) :  {:12.6f}".format(delta_tP2))
        print("time phase 3 (s) :  {:12.6f}".format(delta_tP3))
        return complete_pdag



if __name__ == "__main__":
     
    N_sample = params.N
    random_state = params.rstate
    output_dir = params.output_dir
    input = params.input
    file_adjlist = params.file_adjlist
    file_edgelist = params.file_edgelist
    

    #reading input data set into pandas dataframe
 

    data_TNG = pd.read_csv(input)
    
    variables= data_TNG.columns.to_list()

    #data_TNG = df_TNG300.compute()


    #RESHUFFLING DATA
    data_TNG = data_TNG.sample(frac=1, random_state=random_state).reset_index()

    data_TNG = data_TNG.head(N_sample)
    

    #iCI_test = "dcc_gamma"
    iCI_test = "hsic_gamma"
    
    
    ts= time.perf_counter() 


    
    final_pdag = estimate(data_TNG, variables, ici_test=iCI_test,  random_seed= random_state,significance_level=0.05)
    ts2 = time.perf_counter() - ts
    print("Elapsed time PC (sec) :  {:12.6f}".format(ts2))

    print(final_pdag.edges())

    
    G_last = nx.DiGraph()

    G_last.add_edges_from( final_pdag.edges())  

    plot(G_last, methodtype='graphviz', filename="base_DAG.png")  

    nx.write_adjlist(G_last, file_adjlist)

    nx.write_edgelist(G_last,file_edgelist)

    


