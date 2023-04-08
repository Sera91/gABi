#!/usr/bin/env python

import logging
from itertools import chain, combinations, permutations
import numpy as np
import networkx as nx
import pandas as pd
import pickle
import dask
import dask.dataframe as dd
import os
import gc
import sys
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from coreBN.base import PDAG
from coreBN.estimators import StructureEstimator
from coreBN.CItests import kernel_CItest, kernel_CItest_cycle
from coreBN.global_vars import SHOW_PROGRESS
from pygam import LinearGAM, s
from coreBN.CItests import Hsic_gamma_py
#from coreBN.CItests.hsic_perm import Hsic_perm_or
from coreBN.CItests import Dcov_gamma_py 
#from coreBN.CItests.dcc_perm import Dcov_perm_or
from dask.distributed import as_completed
#from dask.distributed import Client
import joblib

class kPC(StructureEstimator):
    def __init__(self, data=None, independencies=None, **kwargs):
        """
        Class for constraint-based estimation of DAGs using the PC algorithm
        from a given data set.
        This Estimatiior identifies (conditional) dependencies in data
        set using kernel based conditional dependency test and uses the PC algorithm to
        estimate a DAG pattern that satisfies the identified dependencies. 
        The DAG pattern can then be completed to a faithful DAG, if possible,
        fixing undirected edges.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.  (If some
            values in the data are missing the data cells should be set to
            `numpy.NaN`.)

        References
        ----------
        [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques,
            2009, Section 18.2
        [2] Neapolitan, Learning Bayesian Networks, Section 10.1.2 for the PC algorithm (page 550), http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
        """
        super(kPC, self).__init__(
            data=data, **kwargs)

    def estimate(
        self,
        variant="stable",
        ci_test="dcc_gamma",
        N_obs=2000,
        random_seed=1,
        dask_client = None ,
        return_type="dag",
        significance_level=0.01,
        n_jobs=-1,
        show_progress=True,
        **kwargs,
    ):
        """
        Estimates a DAG/PDAG from the given dataset using the kernel PC algorithm which
        is a generalization of the PC algorithm, the most common constraint-based structure learning algorithm[1].
        The conditional independencies among the variables in the dataset are identified by doing statistical independece test. 
        This method returns a DAG/PDAG structure which is faithful to the independencies
        implied by the dataset.

        It can be applied to continuous vars with priors

        Parameters
        ----------
        variant: str ("stable", "parallel")
            The variant of PC algorithm to run.
            "orig": The original PC algorithm. Might not give the same
                    results in different runs but does less independence
                    tests compared to stable.
            "stable": Gives the same result in every run but does needs to
                    do more statistical independence tests.
            "parallel": Parallel version of PC Stable. Can run on multiple
                    cores with the same result on each run.

        ci_test: str or fun
            The statistical tests to use for testing conditional independence in
            the dataset of continuous non gaussian vars are:
            -"HSIC-GAMMA"
            -"HSIC-PERM"
            -"DCOV-GAMMA"
            -"DCOV-PERM"


        return_type: str (one of "dag", "cpdag", "pdag", "skeleton")
            The type of structure to return.

            If `return_type=pdag` or `return_type=cpdag`: a partially directed structure is returned.
            If `return_type=dag`, a fully directed structure is returned if it
                is possible to orient all the edges.
            If `return_type="skeleton", returns an undirected graph along
                with the separating sets.

        significance_level: float (default: 0.01)
            The statistical tests use this value to compare with the p-value of
            the test to decide whether the tested variables are independent or
            not.
            For all the tests: 
            if p-value > significance_level, it assumes that the
                    independence condition satisfied in the data.


        Returns
        -------
        Estimated model: coreBN.base.DAG, coreBN.base.PDAG, or tuple(networkx.UndirectedGraph, dict)
                The estimated model structure, can be a partially directed graph (PDAG)
                or a fully directed graph (DAG), or (Undirected Graph, separating sets)
                depending on the value of `return_type` argument.

        References
        ----------
        [1] Original PC: P. Spirtes, C. Glymour, and R. Scheines, Causation,
                    Prediction, and Search, 2nd ed. Cambridge, MA: MIT Press, 2000.
        [2] Stable PC:  D. Colombo and M. H. Maathuis, “A modification of the PC algorithm
                    yielding order-independent skeletons,” ArXiv e-prints, Nov. 2012.
        [3] Parallel PC: Le, Thuc, et al. "A fast PC algorithm for high dimensional causal
                    discovery with multi-core PCs." IEEE/ACM transactions on computational
                    biology and bioinformatics (2016).
        [4] Kernel-PC: Verbyla (2017)

        [5] Kernel-PC : Gretton et al. (2009-2015)

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from coreBN.estimators import kPC
        >>> data = load_dataframe("kpc_basic_example")

        >>> c = kPC(data)
        >>> model = c.estimate()
        >>> print(model.edges())

        """
        # Step 0: Do checks that the specified parameters are correct, else throw meaningful error.
        if variant not in ("orig", "stable", "parallel", "dask_parallel", "dask_backend"):
            raise ValueError(
                f"variant must be one of: orig, stable, parallel or dask_parallel. Got: {variant}"
            )
        if (not callable(ci_test)) and (ci_test not in ("hsic_gamma", "hsic_perm", "dcc_perm", "dcc_gamma")):
            raise ValueError(
                "ci_test must be a callable or one of: hsic_gamma, hsic_perm, dcc_perm, dcc_gamma"
            )

        if (ci_test in ("hsic_gamma", "hsic_perm", "dcc_perm", "dcc_gamma")) and (self.data is None):
            raise ValueError(
                "For using kernel CI test data argument must be specified"
            )

        # Step 1: Run the PC algorithm to build the skeleton and get the separating sets.
        skel, separating_sets = self.build_skeleton(ci_test, dask_client, N_obs, random_seed, significance_level=significance_level,
                                                    variant=variant, n_jobs=n_jobs, show_progress=show_progress, **kwargs,)

        if return_type.lower() == "skeleton":
            return skel, separating_sets

        print("AFTER STEP 1:")
        print("list nodes in skeleton:", skel.nodes())
        print("list edges in skeleton:", skel.edges())

        # Step 2: Find V structures/Immoralities
        pdag = self.skeleton_to_pdag(skel, separating_sets)

        print("PRINT after call to skeleton to PDAG")
        print("list nodes: ", pdag.nodes())
        print("list directed edges: ", pdag.directed_edges)
        print("list undirected edges: ", pdag.undirected_edges )


        # Step 3: Generalized transitive phase

        pdag = self.regrVonPS(pdag, ci_test=ci_test,
                              significance_level=0.06)

        # Step 4: Applying Meek's rules

        complete_pdag = self.PDAG_by_Meek_rules(pdag)

        # Step 5: Either return the CPDAG or fully orient the edges to build a DAG.
        if return_type.lower() in ("pdag", "cpdag"):
            return complete_pdag
        elif return_type.lower() == "dag":
            return complete_pdag.to_dag()
        else:
            raise ValueError(
                f"return_type must be one of: dag, pdag, cpdag, or skeleton. Got: {return_type}"
            )

    def build_skeleton(
        self,
        ci_test="hsic_gamma",
        dask_client = None,
        N_sample=2000,
        rstate=1,
        significance_level=0.05,
        variant="stable",
        n_jobs=-1,
        show_progress=True,
        **kwargs,
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
        elif ci_test == "hsic_perm":
            sel_method="hsic.perm"
        elif ci_test == "dcc_perm":
            sel_method="dcc.perm"
        elif ci_test == "dcc_gamma":
            sel_method="dcc.gamma"
        else:
            raise ValueError(
                f"ci_test must be either hsic_gamma, hsic_perm or dcc_perm or a function. Got: {ci_test}")
        variables = self.variables[1:]
        
        max_cond_vars= len(variables)-2
        
        
        print("I have created the moral graph")
        # Step 1: Initialize a fully connected undirected graph
        graph = nx.complete_graph(n=variables, create_using=nx.Graph)

        print("I am reading the data")

        
        #if self.data_input is not None:
        #    if os.path.isdir(self.data_input):
        #        data = dd.read_parquet(self.data_input).head(10000)
        #        data.sample(frac=1, random_state=rstate).reset_index()
        #        data = data.head(N_sample)
        #    else:
        #        data = dd.read_csv(self.data_input)
        #        data.sample(frac=1, random_state=rstate).reset_index()
        #        data = data.head(N_sample)
        if self.data is not None:
           data=self.data
           print("variables in data:", variables)
        else:
            print('error: missing input data')
            sys.exit()

        if show_progress and SHOW_PROGRESS:
            pbar = tqdm(total=max_cond_vars)
            pbar.set_description("Starting working")

        if (dask_client!=None) and (variant == "dask_parallel"):

            dask_client.scatter(data)
        
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

                futures = dask_client.map(parallel_CItest, list_sel_edges, list_sepsets, list_limns, list_n_devices)

		
                for future, result in as_completed(futures, with_results=True):
                    list_results.append(result)
			
                for result in list_results:
                   if result[0] != None:
                        u, v = result[0]
                        separating_sets[frozenset((u,v))]= result[1]
                        graph.remove_edge(u,v)

                file_dictionary='sepsets-TNG300-seldisks-at-step'+str(lim_neighbors)+'_rstate'+str(rstate)+'_allvars_2n.pkl'
                
                with open(file_dictionary, 'wb') as fp:
                    pickle.dump(separating_sets, fp)
                    print('dictionary saved successfully to file')
                
                # Step 3: After iterating over all the edges, expand the search space by increasing the size
                #         of conditioning set by 1.
              
                del file_dictionary, list_limns, list_n_devices, list_sepsets, neighbors
                gc.collect()
                lim_neighbors += 1
                if show_progress and SHOW_PROGRESS:
                    pbar.update(1)
                    pbar.set_description(
                        f"Working for n conditional variables: {lim_neighbors}")

            
            
        if  (dask_client!=None) and (variant == "dask-backend"): 
            lim_neighbors = 0
            separating_sets = dict()
            # Exit condition: 1. If all the nodes in graph has less than `lim_neighbors` neighbors.
            #             or  2. `lim_neighbors` is greater than `max_conditional_variables`.
            while not all(
                [len(list(graph.neighbors(var))) <
                 lim_neighbors for var in variables]
            ):

                def _dask_parallel_fun(u, v):
                    for separating_set in chain(
                        combinations(set(graph.neighbors(u)) -
                                     set([v]), lim_neighbors),
                        combinations(set(graph.neighbors(v)) -
                                     set([u]), lim_neighbors),
                    ):
			
                        if kernel_CItest(
                            u,
                            v,
                            separating_set,
                            data_input=self.data,
                            method=sel_method,
                            boolean=True,
                            significance_level=significance_level,
                            **kwargs,
                        ):
                            return (u, v), separating_set

                # apply dask backend
                #client = Client(dask_cluster)
                # We expect SLURM_NTASKS-2 workers
                N = int(os.getenv('SLURM_NTASKS'))-2

                # Wait for these workers and report
                dask_client.wait_for_workers(n_workers=N)

                num_workers = len(dask_client.scheduler_info()['workers'])
                print("%d workers available and ready" % num_workers)

                if n_jobs > N:
                    n_jobs = N

                with joblib.parallel_backend('dask'):
                    results = Parallel(n_jobs=n_jobs)(joblib.delayed(_dask_parallel_fun)(u, v, self.data) for (u, v) in graph.edges()
                                                                        )
                for result in results:
                    if result is not None:
                        (u, v), sep_set = result
                        graph.remove_edge(u, v)
                        separating_sets[frozenset((u, v))] = sep_set

                # Step 3: After iterating over all the edges, expand the search space by increasing the size
                #         of conditioning set by 1.
                if lim_neighbors >= max_cond_vars:
                    logging.info(
                            "Reached maximum number of allowed conditional variables. Exiting"
                    )
                    break
                lim_neighbors += 1
                print(f"Working for n conditional variables: {lim_neighbors}")

                if show_progress and SHOW_PROGRESS:
                    pbar.update(1)
                    pbar.set_description(
                        f"Working for n conditional variables: {lim_neighbors}"
                    )

        else:

            lim_neighbors = 0
            separating_sets = dict() #dictionary where we will save each separating set for pair of nodes

            # Exit condition: 1. If all the nodes in graph has less than `lim_neighbors` neighbors.
            #             or  2. `lim_neighbors` is greater than `max_conditional_variables`.
            while not all(
                [len(list(graph.neighbors(var))) < lim_neighbors for var in variables]):

                # Step 2: Iterate over the edges and find a conditioning set of
                # size `lim_neighbors` which makes u and v independent.
                if variant == "orig":
                    for (u, v) in graph.edges():
                        for separating_set in chain(
                            combinations(set(graph.neighbors(u)) -
                                         set([v]), lim_neighbors),
                            combinations(set(graph.neighbors(v)) -
                                         set([u]), lim_neighbors),
                        ):
                            # If a conditioning set exists remove the edge, store the separating set
                            # and move on to finding conditioning set for next edge.
                            if kernel_CItest(
                                u,
                                v,
                                separating_set,
                                data,
                                method=sel_method,
                                boolean=True,
                                significance_level=significance_level,
                                **kwargs,
                            ):
                                separating_sets[frozenset(
                                    (u, v))] = separating_set
                                graph.remove_edge(u, v)
                                break

                elif variant == "stable":
                    print("I am in variant stable")
                    # In case of stable, precompute neighbors as this is the stable algorithm.
                    neighbors = {node: set(graph[node]) for node in graph.nodes()}
                    for (u, v) in graph.edges():
                        for separating_set in chain(
                            combinations( set(neighbors[u]) - set([v]), lim_neighbors),
                            combinations( set(neighbors[v]) - set([u]), lim_neighbors),
                        ):
                            # If a conditioning set exists remove the edge, store the
                            # separating set and move on to finding conditioning set for next edge.

                            all_vars = list([u,v])
                            all_vars = all_vars + list(separating_set)
                            data_sel = data[all_vars]
                            print("I am going to do the test")
                            if kernel_CItest(
                                u,
                                v,
                                separating_set,
                                data_sel,
                                method=sel_method,
                                boolean=True,
                                significance_level=significance_level,
                                **kwargs,
                            ):
                                separating_sets[frozenset((u, v))] = separating_set
                                graph.remove_edge(u, v)
                                break
                            del data_sel

                elif variant == "parallel":
                    neighbors = {node: set(graph[node])
                                 for node in graph.nodes()}

                    def _parallel_fun(u, v):
                        for separating_set in chain(
                            combinations(set(graph.neighbors(u)) -
                                         set([v]), lim_neighbors),
                            combinations(set(graph.neighbors(v)) -
                                         set([u]), lim_neighbors),
                        ):
                            if kernel_CItest(
                                u,
                                v,
                                separating_set,
                                data=data,
                                method=sel_method,
                                independencies=self.independencies,
                                significance_level=significance_level,
                                **kwargs,
                            ):
                                return (u, v), separating_set

                    results = Parallel(n_jobs=n_jobs, prefer="threads")(
                        delayed(_parallel_fun)(u, v) for (u, v) in graph.edges()
                    )
                    for result in results:
                        if result is not None:
                            (u, v), sep_set = result
                            graph.remove_edge(u, v)
                            separating_sets[frozenset((u, v))] = sep_set

                else:
                    raise ValueError(
                        f"variant must be one of (orig, stable, parallel, dask_parallel). Got: {variant}"
                    )

                # Step 3: After iterating over all the edges, expand the search space by increasing the size
                #         of conditioning set by 1.
                if lim_neighbors >= max_cond_vars:
                    logging.info(
                        "Reached maximum number of allowed conditional variables. Exiting"
                    )
                    break
                lim_neighbors += 1

                if show_progress and SHOW_PROGRESS:
                    pbar.update(1)
                    pbar.set_description(
                        f"Working for n conditional variables: {lim_neighbors}"
                    )

        if show_progress and SHOW_PROGRESS:
            pbar.close()
        print("EXITING from build_skeleton")
        return graph, separating_sets

    @staticmethod
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
        Model after edge orientation: coreBN.base.DAG
            An estimate for the DAG pattern of the BN underlying the data. The
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
        >>> from coreBN.estimators import PC
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
                for Z in set(skeleton.neighbors(X)) & set(skeleton.neighbors(Y)):
                    if Z not in separating_sets[frozenset((X, Y))]:
                        pdag.remove_edges_from([(Z, X), (Z, Y)])

        # TODO: This is temp fix to get a PDAG object.
        edges = set(pdag.edges())
        undirected_edges = []
        directed_edges = []
        for u, v in edges:
            if (v, u) in edges:
                undirected_edges.append((u, v))
            else:
                directed_edges.append((u, v))
        return PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)

    def regrVonPS(
        self,
        pdag,
        ci_test="hsic_gamma",
        significance_level=0.06
    ):
        """
        uses the genralised additive model to non-linearly
        and non-parametrically regress variable V on the vars
        present in its neighborood, to test residuals  .
        """
        if ci_test == "hsic_gamma":
            sel_method="hsic.gamma"
        elif ci_test == "hsic_perm":
            sel_method="hsic.perm"
        elif ci_test == "dcc_perm":
            sel_method="dcc.perm"
        elif ci_test == "dcc_gamma":
            sel_method="dcc.gamma"
        else:
            raise ValueError(
                f"ci_test must be either hsic_gamma, hsic_perm or dcc_perm or a function. Got: {ci_test}")

    
        #data = pd.read_csv(self.data_input)

        #if os.path.isdir(self.data_input):
        #        data = dd.read_parquet(self.data_input).head(2000)
        #else:
        #        data = dd.read_csv(self.data_input).head(2000)
        data=self.data
        print("selected method for CI test is:", sel_method)
        residuals_matrix = np.zeros(data.shape, dtype=np.float64)
        # I will use a mapping array to convert string into col index
        # for the residuals matrix

        # this can be avoided defining a recarray in the form:
        #residual_recarray= np.zeros(self.data.shape[0], dtype=list(zip(list(graph.nodes), [np.float64]*self.data.shape[1])))

        for i, node in enumerate(pdag.nodes()):
            print('estimating residuals of var:', node)
            
            print('nodes in neighborhood[',node,']:', list(pdag.neighbors(node)))
            print('parents [',node, ']', list(pdag.predecessors(node)))
            print('children[',node, ']', list(pdag.successors(node)))
            set_Z = list(pdag.predecessors(node))
            if len(set_Z)!=0:
                gam = LinearGAM(np.sum([s(ii) for ii in range(len(set_Z))]))
                y_arr = (data[node]).to_numpy()
                data_Sset = (data[set_Z]).to_numpy()
                gam.gridsearch(data_Sset, y_arr)
                residuals_matrix[:, i] = gam.deviance_residuals(data_Sset, y_arr)
            else:
                print("this node has no possible predecessors")

        print("nodes in pdag:", list(pdag.nodes()))
        print("all edges in pdag:",list(pdag.edges()))
        print("undirected edges in pdag", list(pdag.undirected_edges))

        screened_edges=[]
        arr_names_nodes = np.array(list(pdag.nodes()))
        for l,r in pdag.undirected_edges:
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

            index=1
            p=100
            sigma = 1.0
            numCol = 100


            match sel_method:
                case 'dcc.gamma':
                  p_left = Dcov_gamma_py(x_left, y_left, 0)
                  p_right = Dcov_gamma_py(x_right, y_right, 0)
                
                case 'hsic.gamma':
                  p_left = Hsic_gamma_py(x_left, y_left, 0)
                  p_right = Hsic_gamma_py(x_right, y_right, 0)
		#case 'hsic.perm':
                  #p_left = Hsic_perm_or(x_left, y_left, sigma, p, numCol)
                  #p_right = Hsic_perm_or(x_right, y_right, sigma, p, numCol)
                #case 'dcc.perm':    
                  #p_left = Dcov_perm_or(x_left, y_left, index, p)
                  #p_right = Dcov_perm_or(x_right, y_right, index, p)
                    
            
            if ((p_left > significance_level) & (p_right < significance_level)):
                # remove edge from undirected
                # leaves edge right-> left as directed
                pdag.remove_edge(left_vert, right_vert)
            if ((p_right > significance_level) & (p_left < significance_level)):
                # remove edge from undirected
                # leaves edge lef t-> right as directed
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

    @staticmethod
    def PDAG_by_Meek_rules(pdag):

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
                for Z in (
                    set(pdag.successors(X))
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
        edges = set(pdag.edges())
        undirected_edges = []
        directed_edges = []
        for u, v in edges:
            if (v, u) in edges:
                undirected_edges.append((u, v))
            else:
                directed_edges.append((u, v))
        return PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)
