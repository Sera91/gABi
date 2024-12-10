import numpy as np
import pandas as pd
import pickle
import sys
import os
import gc
import time
import networkx as nx
#import matplotlib.pyplot as plt
from itertools import chain, combinations, permutations
from coreBN.utils import GAM_residuals, GAM_residuals_fast
#import coreBN
from coreBN.CItests import kernel_CItest_cycle
from hsic_gamma_pytorch_h import Hsic_gamma_py
from dcc_gamma_pytorch_h import Dcov_gamma_py_h
from coreBN.base import PDAG
from dask_mpi import initialize
from dask.distributed import Client, as_completed
import dask.dataframe as dd
import params_basic as params





def kernel_CItest( x, y, list_set, n_device, data, method='dcc.gamma', p=None, index=1, sig=1, numCol=None, verbose=False):
    
        """
        This function tests whether x and y are conditionally independent given the subset S of the remaining nodes,
        using the independence criterions: Distance Covariance/HSIC
        It takes as arguments:
                 @dataframe :  data
                 @str       : x,y (identify vars over whic we test CI, in the dataframe)
                 @list of str: list_set  (list of names identifying vars in the conditioning set)
                 @bool param:  verbose (a logical parameter, if None it is setted to False. When True the detailed output is provided).
                 @str  param: method  (Method for the conditional independence test: Distance Covariance (permutation or gamma test), HSIC (permutation or gamma test) or HSIC cluster)
                 @int  param: p  (number of permutations for Distance Covariance, HSIC permutation and HSIC cluster tests)
                 @int  param: index (power index in (0,2]  for te formula of the distance in the Distance Covariance)
                 @float param: sig (Gaussian kernel width for HSIC tests. Default is 1)
                 @int  param: numCol (number of columns used in the incomplete Cholesky decomposition. Default is 100)
                 @int  param: numCluster (number of clusters for kPC clust algorithm)
                 @float param: eps   (Normalization parameter for kPC clust. Default is 0.1)
                 test returns the  p_value         
        """

        #if dask_cluster!=None:
        #    client = Client(dask_cluster)
        #print("X variable: ", x)
        #print("Y variable:", y)

        #data = pd.read_csv(data_input)
        #reading parquet files
        #ddf = dd.read_parquet(data_input)
        #all_vars = list(list_set)
        #all_vars.append(x)
        #all_vars.append(y)

        #data = ddf[all_vars].compute()
        #data = data.head(1000)
        x_arr = (data[x]).to_numpy()
        y_arr = (data[y]).to_numpy()
                 
        #if(boolean==True):
        #    print("significance level:", kwargs["significance_level"])    
        #if debug:
        #    print("Independence criterion method was not provided, using the default method: hsic-gamma")
        if (p==None):
                p = 100
                #print("Number of perm not provided. Default is 100")
        #if (index==None):
        #        index=1
                #print("index for Dcov not provided. default is 1")
        #if (sig==None):
        #        sig=1
                #print("Gaussian kernel width for HSIC tests not provided. Default is 1")
        if (numCol==None):
                #print("Number of cols to consider in Cholesky decomposition not provided. Default is 100")
                numCol=100

        #p_value=0.0
        N_cond_vars = len(list_set)
        if (N_cond_vars<1):
                    print("pure independence test")
                    final_x_arr = x_arr
                    final_y_arr = y_arr
        else :
                    
                    list_set = list(list_set)
                    print("conditioning set:" + str(list_set), flush=True)
                    #print(type(list_set))
                    data_Sset = (data[list_set]).to_numpy()
                    res_X = GAM_residuals(data_Sset, x_arr, N_cond_vars)
                    res_Y =  GAM_residuals(data_Sset, y_arr, N_cond_vars)
                    final_x_arr = res_X
                    final_y_arr = res_Y
                    del data_Sset
                         
                    
        #match method:
        #    case 'dcc.perm':    
                  
        if method =='dcc.gamma':    
                  #NEED  to introduce flag for backend here
                  p_value = Dcov_gamma_py_h(final_x_arr, final_y_arr, n_device)
                  #p_value = Dcov_gamma_py_gpu(final_x_arr, final_y_arr, index)
        elif method == 'hsic.gamma':
                  p_value = Hsic_gamma_py(final_x_arr, final_y_arr , sig)
        else:
                  print('wrong given method')	
                  sys.exit()
        
        print('pval:', p_value)
        return p_value


def build_skeleton(
        data,
        variables,
        ci_test,
	    client,
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
        
        max_cond_vars= len(variables) - 2
        
        if verbose:
            print("I have created the moral graph")
        # Step 1: Initialize a fully connected undirected graph
        graph = nx.complete_graph(n=variables, create_using=nx.Graph)

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

        client.scatter(data)
        
        def parallel_CItest(edge, sep_sets_edge,lim_n, n_device):
                        u,v = edge
                        print("Kernel CI test pair(", u, v, ")", flush=True)
                        edge, sep_set, p_value = kernel_CItest_cycle(u,v,sep_sets_edge, lim_n, significance_level, n_device, data, method=sel_method )
                        # If a conditioning set exists stores the edge to remove, store the
                        # separating set and move on to finding conditioning set for next edge
                        if ( p_value > significance_level) :
                                               return ([edge, sep_set])
                        return np.array([None, None])

        lim_neighbors = 0
        
        while not all([len(list(graph.neighbors(var))) < lim_neighbors for var in variables]) and lim_neighbors <= max_cond_vars:

                # Step 2: Iterate over the edges and find a conditioning set of
                # size `lim_neighbors` which makes u and v independent.
                 
                neighbors = {node: set(graph[node]) for node in graph.nodes()}
                list_edges = list(graph.edges())
                print(("edges at step"+str(lim_neighbors)+":"), list_edges)
                list_results=list([])	
                if (lim_neighbors > 0):
                    list_sel_edges=[(u,v) for (u,v) in list_edges if (len(list(neighbors[u]-set([v])))>=lim_neighbors or len(list(neighbors[v]-set([u])))>=lim_neighbors)]
                else:
                    list_sel_edges=list_edges
                list_n_devices= np.zeros(len(list_sel_edges), dtype = np.int32)
                list_n_devices[0::2] = 0
                list_n_devices[1::2] = 1
                list_n_devices = list_n_devices.tolist()
                #list_sel_edges = list([(u,v) for (u,v) in list_edges if ((len(list(graph.neighbors(u)))>0) or (len(list(graph.neighbors(v)))>0)) ])
                list_sepsets = list([set(list(chain(combinations( set(neighbors[u]) - set([v]), lim_neighbors), combinations( set(neighbors[v]) - set([u]), lim_neighbors),))) for (u,v) in list_sel_edges])
                

                list_limns =( np.zeros(len(list_sel_edges), dtype = np.int32) + lim_neighbors).tolist()

                futures = client.map(parallel_CItest, list_sel_edges, list_sepsets, list_limns, list_n_devices)

		
                for future, result in as_completed(futures, with_results=True):
                    list_results.append(result)
			
                for result in list_results:
                   if result[0] != None:
                        u, v = result[0]
                        separating_sets[frozenset((u,v))]= result[1]
                        graph.remove_edge(u,v)

                #file_dictionary='sepsets-TNG300-seldisks-at-step'+str(lim_neighbors)+'_rstate'+str(rstate)+'_allvars_4n.pkl'
                
                #with open(file_dictionary, 'wb') as fp:
                #    pickle.dump(separating_sets, fp)
                #    print('dictionary saved successfully to file')
                
                # Step 3: After iterating over all the edges, expand the search space by increasing the size
                #         of conditioning set by 1.
              
                #del file_dictionary
                del list_limns, list_n_devices, list_sepsets, neighbors
                gc.collect()
                lim_neighbors += 1

               
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
                for Z in set(pdag.neighbors(X)):
                    if (Z in set(pdag.neighbors(Y))) and (Z not in separating_sets[frozenset((X, Y))]):
                        pdag.remove_edges_from([(Z, X), (Z, Y)])

        # TODO: This is temp fix to get a PDAG object.
        edges = pdag.edges()
        undirected_edges = []
        directed_edges = []
        for edge in edges:
            u, v = edge
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
            sel_method="hsic.gamma"
        elif ci_test == "dcc_gamma":
            sel_method="dcc.gamma"
        else:
            raise ValueError(
                f"ci_test must be either hsic_gamma, dcc_gamma, hsic_perm or dcc_perm or a function. Got: {ci_test}")

    
        #data = pd.read_csv(self.data_input)
        if verbose:
            print("selected method for CI test is:", sel_method)
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
                residuals_matrix[:, i] = GAM_residuals(data_Sset, y_arr, S_Z)
            else:
                print("this node has no possible predecessors")

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

            #index=1
            #p=100
            #sigma = 1.0
            #numCol = 100


            match sel_method:
                case 'dcc.gamma':    
                  p_left = Dcov_gamma_py_h(x_left, y_left,0)
                  p_right = Dcov_gamma_py_h(x_right, y_right, 0)
                case 'hsic.gamma':
                  p_left = Hsic_gamma_py(x_left, y_left,0)
                  p_right = Hsic_gamma_py(x_right, y_right,0)
            
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

def estimate(data_TNG, variables, ci_test, input_client,random_seed, significance_level=0.05, debug=True, timing=True):
        if debug:
            print('I am building the graph skeleton')
        # Step 1: Run the PC algorithm to build the skeleton and get the separating sets.	
        t_startP1 = time.perf_counter()
        skel, separating_sets = build_skeleton(data_TNG, variables, ci_test,client=input_client, rstate=random_seed, significance_level=0.05)
        t_endP1 = time.perf_counter()
        if debug:
            print("AFTER STEP 1:")
            print("list nodes in skeleton:", skel.nodes())
            print("list edges in skeleton:", skel.edges())

        # Step 2: Find V structures/Immoralities
        t_startP2 = time.perf_counter()	
        pdag = skeleton_to_pdag(skel, separating_sets)
        t_endP2 = time.perf_counter()
        print("PRINT after call to skeleton_to_PDAG")
        print("list directed edges: ", pdag.edges())
        if debug:
            print("list nodes: ", pdag.nodes())
            print("list directed edges: ", pdag.directed_edges)
            print("list undirected edges: ", pdag.undirected_edges )


        # Step 3: Generalized transitive phase
        t_startP3 = time.perf_counter()
        pdag_new = regrVonPS(data_TNG, pdag, ci_test, significance_level=0.06)
        #N_vars= len(df_basic.columns)

        # Step 4: Applying Meek's rules

        complete_pdag = PDAG_by_Meek_rules(pdag_new)
        delta_tP3 = time.perf_counter() -t_startP3
        delta_tP1 = t_endP1- t_startP1
        delta_tP2 = t_endP2- t_startP2 
        if timing:
            print("time build skeleton (s) :  {:12.6f}".format(delta_tP1))
            print("time phase 2 (s) :  {:12.6f}".format(delta_tP2))
            print("time gen. phase 3 (s) :  {:12.6f}".format(delta_tP3))
        return complete_pdag



if __name__ == "__main__":
     
    N_sample = params.N
    random_state = params.rstate
    output_dir = params.output_dir
    input = params.input
    file_adjlist = params.file_adjlist
    file_edgelist = params.file_edgelist
    

    

    #reading input data set into pandas dataframe
    #cluster = LocalCluster()
    #client = Client(cluster)file_adjlist
    #dask_client = Client(processes=False)
    

    print('I am before client inizialization')
    # Initialise Dask cluster and client interface

    n_tasks = int(os.getenv('SLURM_NTASKS'))
    mem = os.getenv('SLURM_MEM_PER_CPU')
    mem = str(int(mem))+'MB'

    initialize(memory_limit=mem)

    dask_client = Client()

    dask_client.wait_for_workers(n_workers=(n_tasks-2))
    #dask_client.restart()

    num_workers = len(dask_client.scheduler_info()['workers'])
    print("%d workers available and ready"%num_workers)
    #df_basic = pd.read_csv(input_file)

    data_TNG = pd.read_csv(input)
    
    variables= data_TNG.columns.to_list()

    #data_TNG = df_TNG300.compute()


    #RESHUFFLING DATA
    data_TNG = data_TNG.sample(frac=1, random_state=random_state).reset_index()

    data_TNG = data_TNG.head(N_sample)
    

    #iCI_test = "dcc_gamma"
    iCI_test = "hsic_gamma"
    
    
    ts= time.perf_counter() 
    
    final_pdag = estimate(data_TNG, variables, ci_test=iCI_test, input_client=dask_client, random_seed= random_state,significance_level=0.05)
    ts2 = time.perf_counter() - ts
    print("Elapsed time [s] kernel-PC with test {} :  {:12.6f}".format(iCI_test,ts2))

    print(final_pdag.edges())

    
    G_last = nx.DiGraph()

    G_last.add_edges_from( final_pdag.edges())    

    nx.write_adjlist(G_last, file_adjlist)

    nx.write_edgelist(G_last,file_edgelist)

    dask_client.shutdown()


