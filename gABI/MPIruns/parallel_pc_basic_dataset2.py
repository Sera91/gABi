import numpy as np
import pandas as pd
from math import sqrt, log
import pickle
import sys
import os
import gc
import time
import networkx as nx
from scipy import stats
from scipy.stats import norm
#import matplotlib.pyplot as plt
from itertools import chain, combinations, permutations
#import coreBN
#from coreBN.CItests import CItest_cycle
from coreBN.base import PDAG
import params_pc_basic as params
import os, json, codecs, time, hashlib
from collections.abc import Iterable
from dask_mpi import initialize
from dask.distributed import Client, as_completed
import dask.dataframe as dd



NO_SPECIFIED_PARAMETERS_MSG = "NO SPECIFIED PARAMETERS"
fisherz = "fisherz"
mv_fisherz = "mv_fisherz"
mc_fisherz = "mc_fisherz"


class Test_Base(object):
    # Base class for CIT, contains basic operations for input check and caching, etc.
    def __init__(self, data, cache_path=None, **kwargs):
        '''
        Parameters
        ----------
        data: data matrix, np.ndarray, in shape (n_samples, n_features)
        cache_path: str, path to save cache .json file. default as None (no io to local file).
        kwargs: for future extension.
        '''
        #vars = data.columns.to_list()
        self.vars = data.columns.to_list()
        data = data.to_numpy()

        assert isinstance(data, np.ndarray), "Input data must be a numpy array."
        self.data = data      
        self.data_hash = hashlib.md5(str(data).encode('utf-8')).hexdigest()
        self.sample_size, self.num_features = data.shape
        self.cache_path = cache_path
        self.SAVE_CACHE_CYCLE_SECONDS = 30
        self.last_time_cache_saved = time.time()
        self.pvalue_cache = {'data_hash': self.data_hash}
        if cache_path is not None:
            assert cache_path.endswith('.json'), "Cache must be stored as .json file."
            if os.path.exists(cache_path):
                with codecs.open(cache_path, 'r') as fin: self.pvalue_cache = json.load(fin)
                assert self.pvalue_cache['data_hash'] == self.data_hash, "Data hash mismatch."
            else: os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    def check_cache_method_consistent(self, method_name, parameters_hash):
        self.method = method_name
        if method_name not in self.pvalue_cache:
            self.pvalue_cache['method_name'] = method_name # a newly created cache
            self.pvalue_cache['parameters_hash'] = parameters_hash
        else:
            assert self.pvalue_cache['method_name'] == method_name, "CI test method name mismatch." # a loaded cache
            assert self.pvalue_cache['parameters_hash'] == parameters_hash, "CI test method parameters mismatch."

    def assert_input_data_is_valid(self, allow_nan=False, allow_inf=False):
        assert allow_nan or not np.isnan(self.data).any(), "Input data contains NaN. Please check."
        assert allow_inf or not np.isinf(self.data).any(), "Input data contains Inf. Please check."

    def save_to_local_cache(self):
        if not self.cache_path is None and time.time() - self.last_time_cache_saved > self.SAVE_CACHE_CYCLE_SECONDS:
            with codecs.open(self.cache_path, 'w') as fout: fout.write(json.dumps(self.pvalue_cache, indent=2))
            self.last_time_cache_saved = time.time()

    def get_formatted_XYZ_and_cachekey(self, X, Y, condition_set):
        '''
        reformat the input X, Y and condition_set to
            1. convert to built-in types for json serialization
            2. handle multi-dim unconditional variables (for kernel-based)
            3. basic check for valid input (X, Y no overlap with condition_set)
            4. generate unique and hashable cache key
        Parameters
        ----------
        X: int, or np.*int*, or Iterable<int | np.*int*>
        Y: int, or np.*int*, or Iterable<int | np.*int*>
        condition_set: Iterable<int | np.*int*>
        Returns
        -------
        Xs: List<int>, sorted. may swapped with Ys for cache key uniqueness.
        Ys: List<int>, sorted.
        condition_set: List<int>
        cache_key: string. Unique for <X,Y|S> in any input type or order.
        '''
        from operator import itemgetter

        def _stringize(ulist1, ulist2, clist):
            # ulist1, ulist2, clist: list of ints, sorted.
            _strlst  = lambda lst: '.'.join(map(str, lst))
            return f'{_strlst(ulist1)};{_strlst(ulist2)}|{_strlst(clist)}' if len(clist) > 0 else \
                   f'{_strlst(ulist1)};{_strlst(ulist2)}'

        # every time when cit is called, auto save to local cache.
        self.save_to_local_cache()

        
        if condition_set is None: condition_set = []
       
    

        # usually, X and Y are 1-dimensional index (in constraint-based methods)
       

        dict_ind = { self.vars[i]:i for i in range(0, len(self.vars) ) }
        X= dict_ind[X]
        Y= dict_ind[Y]
        if len(condition_set)==0:
                condition_set=condition_set
        elif len(condition_set)==1:
                print("condition set", condition_set)
                condition_set=dict_ind[condition_set[0]]
                condition_set =[condition_set,]
        else:
                print("condition set", condition_set)
                condition_set= list(itemgetter(*condition_set)(dict_ind))
                condition_set = sorted(set(map(int, condition_set)))
                # 'int' to convert np.*int* to built-in int; 'set' to remove duplicates; sorted for hashing
        
        print("I condition set", condition_set)        
        X, Y = (int(X), int(Y)) if (X < Y) else (int(Y), int(X))
        print('x',X, 'y',Y, flush=True)
        assert X not in condition_set and Y not in condition_set, "X, Y cannot be in condition_set."
        return [X], [Y], condition_set, _stringize([X], [Y], condition_set)



class FisherZ(Test_Base):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('fisherz', NO_SPECIFIED_PARAMETERS_MSG)
        self.assert_input_data_is_valid()
        self.correlation_matrix = np.corrcoef(self.data.T)

    def __call__(self, X, Y, condition_set=None):
        '''
        Perform an independence test using Fisher-Z's test.
        Parameters
        ----------
        X, Y and condition_set : column indices of data
        Returns
        -------
        X : test quantile
        p : the p-value of the test
        '''
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache: return self.pvalue_cache[cache_key]
        var = Xs + Ys + condition_set
        sub_corr_matrix = self.correlation_matrix[np.ix_(var, var)]
        try:
            inv = np.linalg.inv(sub_corr_matrix)
        except np.linalg.LinAlgError:
            raise ValueError('Data correlation matrix is singular. Cannot run fisherz test. Please check your data.')
        r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])
        Z = 0.5 * log((1 + r) / (1 - r))
        X = sqrt(self.sample_size - len(condition_set) - 3) * abs(Z)
        p = 2 * (1 - norm.cdf(abs(X)))
        self.pvalue_cache[cache_key] = p
        return X, p




class MV_FisherZ(Test_Base):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent('mv_fisherz', NO_SPECIFIED_PARAMETERS_MSG)
        self.assert_input_data_is_valid(allow_nan=True)

    def _get_index_no_mv_rows(self, mvdata):
        nrow, ncol = np.shape(mvdata)
        bindxRows = np.ones((nrow,), dtype=bool)
        indxRows = np.array(list(range(nrow)))
        for i in range(ncol):
            bindxRows = np.logical_and(bindxRows, ~np.isnan(mvdata[:, i]))
        indxRows = indxRows[bindxRows]
        return indxRows

    def __call__(self, X, Y, condition_set=None):
        '''
        Perform an independence test using Fisher-Z's test for data with missing values.
        Parameters
        ----------
        X, Y and condition_set : column indices of data
        Returns
        -------
        p : the p-value of the test
        '''
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache: return self.pvalue_cache[cache_key]
        var = Xs + Ys + condition_set
        test_wise_deletion_XYcond_rows_index = self._get_index_no_mv_rows(self.data[:, var])
        assert len(test_wise_deletion_XYcond_rows_index) != 0, \
            "A test-wise deletion fisher-z test appears no overlapping data of involved variables. Please check the input data."
        test_wise_deleted_data_var = self.data[test_wise_deletion_XYcond_rows_index][:, var]
        sub_corr_matrix = np.corrcoef(test_wise_deleted_data_var.T)
        try:
            inv = np.linalg.inv(sub_corr_matrix)
        except np.linalg.LinAlgError:
            raise ValueError('Data correlation matrix is singular. Cannot run fisherz test. Please check your data.')
        r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])
        Z = 0.5 * log((1 + r) / (1 - r))
        X = sqrt(len(test_wise_deletion_XYcond_rows_index) - len(condition_set) - 3) * abs(Z)
        p = 2 * (1 - norm.cdf(abs(X)))
        self.pvalue_cache[cache_key] = p
        return X, p

def CItest_c(data, method='fisherz', **kwargs):
    '''
    Function for Conditional Independence tests on continuous variables
    Parameters
    ----------

    data: numpy.ndarray of shape (n_samples, n_features)
    method: str, in ["fisherz", "mv_fisherz", "gsq"]
    kwargs: placeholder for future arguments, 
    '''
    if method == "fisherz":
        return FisherZ(data, **kwargs)
    elif method == "mv_fisherz":
        return MV_FisherZ(data, **kwargs)
    #elif method == pearsonr:
    #    return Pearsonr(data, **kwargs)
    else:
        raise ValueError("Unknown method: {}".format(method))


def CItest_cycle( x, y, sep_sets, l_m , alpha, data, method='Fisher', verbose=False):

        """
        This function, implemented for the parallel run of pc-stable, tests whether x and y are conditionally independent given all the subsets S, unique combination of the remaining nodes, inside the neighborood of x and y,
        using two tests:
        It takes as arguments:
                 
                 @str       : x,y (identify vars over whic we test CI, in the dataframe)
                 @list of Ssets: list of separating sets, containing each a list of names identifying vars in the conditioning set
                 @l_m       : size of each subset
                 @dataframe :  data
                 @float param: alpha (significance level to test with the p-value test)
                 @int param : integer identifying the cuda device over which perform the GPU calculations
                 @str  param: method  (Method for the conditional independence test: Distance Covariance (permutation or gamma test), HSIC (permutation or gamma test) or HSIC cluster)
                 @int  param: p  (number of permutations for Distance Covariance, HSIC permutation and HSIC cluster tests)
                 @int  param: index (power index in (0,2]  for te formula of the distance in the Distance Covariance)
                 @float param: sig (Gaussian kernel width for HSIC tests. Default is 1)
                 @int  param: numCol (number of columns used in the incomplete Cholesky decomposition. Default is 100)
                 @bool param:  verbose (a logical parameter, if None it is setted to False. When True the detailed output is provided)
                 
        The function returns the  p_value and the corresponding sep_set         
        """


        from operator import itemgetter
        
        #if method =='Pearson':
        #    from .CITests import pearsonr as Itest
        if method == 'Fisher':
            Itest= CItest_c(data, method='fisherz')


        l_sets = list(sep_sets)
        print("first separating set", l_sets[0])
        #print("conditioning sets of len:",N_sets, sep_sets, flush=True)
        
        if (l_m<1):
            print("pure independence test")
            _ , p_value = Itest( X=x, Y=y, condition_set=[])
            return (x,y), {()}, p_value
  
        N_sets = len(l_sets)
        for i in np.arange(0, N_sets):
                l_set=list(l_sets[i])
                _ , p_value = Itest(X=x,Y=y, condition_set=l_set )
                
                #p_value = Dcov_gamma_py_gpu(final_x_arr, final_y_arr, index)
                if (p_value > alpha) :
                    #print(i,'-th pval:', p_value)
                    final_set = l_sets[i]
                    #del data_Sset, data_matrix, x_arr, y_arr, Si_sets, list_vars, l_sets, res_x, res_y
                    #gc.collect()
                    return (x,y), final_set, p_value
        final_set = l_sets[(N_sets-1)]
        #del data_Sset, data_matrix, x_arr, y_arr, Si_sets, list_vars, l_sets, res_x, res_y
        #gc.collect()
        return (x,y), final_set , p_value



def build_skeleton(
        data,
        variables,
        ci_test,
        client,
        rstate,
        significance_level,
        verbose=True
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
        if ci_test == "Pearson":
            sel_method="Fisher"
        elif ci_test == "Fisher":
            sel_method="Fisher"
        else:
            raise ValueError(
                f"ci_test must be either Fisher, Pearson or a function. Got: {ci_test}")
        
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

        client.scatter(data)

        def parallel_CItest(edge, sep_sets_edge,lim_n):
                        u,v = edge
                        print("CI test pair(", u, v, ")", flush=True)
                        edge, sep_set, p_value = CItest_cycle(u,v,sep_sets_edge, lim_neighbors, significance_level, data, method=sel_method )
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
                if verbose:
                    print(("edges at step"+str(lim_neighbors)+":"), list_edges)
                list_results=list([])	
                if (lim_neighbors > 0):
                    list_sel_edges=[(u,v) for (u,v) in list_edges if (len(list(neighbors[u]-set([v])))>=lim_neighbors or len(list(neighbors[v]-set([u])))>=lim_neighbors)]
                else:
                    list_sel_edges=list_edges
                
                
                #list_sel_edges = list([(u,v) for (u,v) in list_edges if ((len(list(graph.neighbors(u)))>0) or (len(list(graph.neighbors(v)))>0)) ])
                list_sepsets = list([set(list(chain(combinations( set(neighbors[u]) - set([v]), lim_neighbors), combinations( set(neighbors[v]) - set([u]), lim_neighbors),))) for (u,v) in list_sel_edges])

                list_limns =( np.zeros(len(list_sel_edges), dtype = np.int32) + lim_neighbors).tolist()

                futures = client.map(parallel_CItest, list_sel_edges, list_sepsets, list_limns)

		
                for future, result in as_completed(futures, with_results=True):
                    list_results.append(result)
			
                for result in list_results:
                   if result[0] != None:
                        u, v = result[0]
                        separating_sets[frozenset((u,v))]= result[1]
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

def estimate(data_TNG, variables, ici_test, input_client, random_seed, significance_level=0.05, debug=True):
        if debug:
            print('I am building the graph skeleton')

        t_startP1 = time.perf_counter() 
        # Step 1: Run the PC algorithm to build the skeleton and get the separating sets.	
        skel, separating_sets = build_skeleton(data_TNG, variables, ici_test,client= input_client, rstate=random_seed, significance_level=0.05)
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
        #pdag_new = regrVonPS(data_TNG, pdag, ici_test, significance_level=0.06)
        #N_vars= len(df_basic.columns)

        # Step 4: Applying Meek's rules

        complete_pdag = PDAG_by_Meek_rules(pdag)
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
    iCI_test = "Fisher"
    
    
    ts= time.perf_counter() 
    
    final_pdag = estimate(data_TNG, variables, ici_test=iCI_test, input_client=dask_client, random_seed= random_state,significance_level=0.05)
    ts2 = time.perf_counter() - ts
    print("Elapsed time PC (sec) :  {:12.6f}".format(ts2))

    print(final_pdag.edges())

    
    G_last = nx.DiGraph()

    G_last.add_edges_from( final_pdag.edges())    

    #nx.write_adjlist(G_last, file_adjlist)

    #nx.write_edgelist(G_last,file_edgelist)

    


