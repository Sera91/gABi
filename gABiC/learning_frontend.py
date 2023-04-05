"""Structure learning. Given a set of data samples, estimate a DAG that captures the dependencies between the variables."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

#from coreBN.estimators import BDeuScore, K2Score, BicScore
#from coreBN.estimators import ExhaustiveSearch, HillClimbSearch

from coreBN.estimators.PC import PC as PC


import gABiC.base as gABiC


# %% Structure Learning
def learn(df, method='pc', black_list=None, white_list=None, bw_list_method=None, test_type='fisher', tabu_length=100, fixed_edges=None, return_all_dags=False, n_jobs=-1, variant="stable", verbose=3):
    """Structure learning function.

    Description
    -----------
    Search strategies for structure learning
    The search space of DAGs is super-exponential in the number of variables and the above scoring functions allow for local maxima.

    To learn model structure (a DAG) from a data set, using constraint-based algorithms there are 2 broad techniques:
            a. PC (based on indep tests)
            b. kernel-PC
        



    Parameters
    ----------
    df : pd.DataFrame()
        Input dataframe.
    method : str, (default : 'pc')
                 Search strategy for structure_learning.
                 It can have the following value
                 'pc' or 'pc-stable'(default)
                 'kpc' or 'kernel-pc' 

    black_list : List or None, (default : None)
                 List of edges that are black listed.
                 or list of nodes to remove from the dataframe. 
                 The resulting model will not contain any nodes that are in black_list.
    white_list : List or None, (default : None)
                 List of edges are white listed.
                 In case of filtering on nodes, the search is limited to those nodes. 
                 The resulting model will then only contain nodes that are in white_list.

                 Works only in case of method='hc' See also paramter: `bw_list_method`
    bw_list_method :  str or tuple, (default : None)
                     It can assume the values:
                     - 'edges' 
                     - 'nodes' 
    
    fixed_edges: iterable, Only in case of HillClimbSearch.
                 A list of edges that will always be there in the final learned model. The algorithm will add these edges at the start of the algorithm and will never change it.
    return_all_dags : Bool, (default: False)
        Return all possible DAGs. Only in case method='exhaustivesearch'
    verbose : int, (default= 3)
              0: None, 1: Error,  2: Warning, 3: Info (default), 4: Debug, 5: Trace

    Returns
    -------
    dict with model.

    Examples
    --------
    >>> # Import gABiC
    >>> import gABiC as sl
    >>> # Load DAG
    >>> model = sl.import_DAG('asia')
    >>> # plot ground truth
    >>> G = sl.plot(model)
    >>> # Sampling
    >>> df = sl.sampling(model, n=10000)
    >>> # Structure learning of sampled dataset
    >>> model_sl = sl.structure.learn(df, method='pc', scoretype='bic')
    >>>
    >>> # Compute edge strength using chi-square independence test
    >>> model_sl = sl.independence_test(model_sl, df)
    >>>
    >>> # Plot based on structure learning of sampled data
    >>> sl.plot(model_sl, pos=G['pos'])
    >>>
    >>> # Compare networks and make plot
    >>> sl.compare_networks(model, model_sl, pos=G['pos'])

    """
    out = []
    # Set config
    config = {'method': method, 'test_type': test_type, 'black_list': black_list, 'white_list': white_list, 'bw_list_method': bw_list_method,  'tabu_length': tabu_length, 'fixed_edges': fixed_edges, 'return_all_dags': return_all_dags, 'n_jobs': n_jobs, 'verbose': verbose}
    # Make some checks
    config = _make_checks(df, config, verbose=verbose)
    # Make sure columns are of type string
    df.columns = df.columns.astype(str)
    # Filter on white_list and black_list
    df = _white_black_list_filter(df, white_list, black_list, bw_list_method=config['bw_list_method'], verbose=verbose)
    # Lets go!
    if config['verbose']>=3: print('[gABiC] >Computing best DAG using [%s]' %(config['method']))

    


    # Constraint-based Structure Learning
    if config['method']=='pc' or config['method']=='pc-stable':
        """PC constraint-based Structure Learning algorithm
        
        Construct DAG (pattern) according to identified independencies between vars, based on (Conditional) Independence Tests
	for Linear Gaussian Models.
        """
        out = _pc_wrapper(df, n_jobs=config['n_jobs'], verbose=config['verbose'], variant=variant)

    elif config['method']=='kpc' or config['method']=='kernel-pc':
        """Kernel-PC constraint-based Structure Learning algorithm
        
        Construct DAG (pattern) according to identified independencies between vars, based on (Conditional) Independence Tesv
	for Non-linear Additive Noise models.
        """
        out = _Kpc_wrapper(df, n_jobs=config['n_jobs'], verbose=config['verbose'], variant=variant)

    else:
        print("given method is not expected. Possible methods for structure learning are pc and kpc")
        sys.exit()

    
    # 
    out['model_edges'] = list(out['model'].edges())
    out['adjmat'] = gABiC.dag2adjmat(out['model'])
    out['config'] = config

    # return
    return(out)


# %% Make Checks
def _make_checks(df, config, verbose=3):
    assert isinstance(pd.DataFrame(), type(df)), 'df must be of type pd.DataFrame()'
    #if not np.isin(config['scoring'], ['bic', 'k2', 'bdeu']): raise Exception('"scoretype=%s" is invalid.' %(config['scoring']))
    if not np.isin(config['method'], ['kpc', 'kernel-pc', 'pc', 'pc-stable']): raise Exception('"method=%s" is invalid.' %(config['method']))

    if isinstance(config['white_list'], str):
        config['white_list'] = [config['white_list']]
    if isinstance(config['black_list'], str):
        config['black_list'] = [config['black_list']]

    if (config['white_list'] is not None) and len(config['white_list'])==0:
        config['white_list'] = None
    if (config['black_list'] is not None) and len(config['black_list'])==0:
        config['black_list'] = None

    if (config['method']!='hc') and (config['bw_list_method']=='edges'): raise Exception('[gABiC] >The "bw_list_method=%s" does not work with "method=%s"' %(config['bw_list_method'], config['method']))
   
    if config['fixed_edges'] is None:
        config['fixed_edges']=set()

    # Show warnings
    if (config['bw_list_method'] is None) and ((config['black_list'] is not None) or (config['white_list'] is not None)):
        raise Exception('[gABiC] >Error: The use of black_list or white_list requires setting bw_list_method.')
    if df.shape[1]>10 and df.shape[1]<15:
        if verbose>=2: print('[gABiC] >Warning: Computing DAG with %d nodes can take a very long time!' %(df.shape[1]))
    return config





# %% white_list and black_list
def _white_black_list_filter(df, white_list, black_list, bw_list_method='edges', verbose=3):
    if bw_list_method=='nodes':
        # Keep only variables that are in white_list.
        if white_list is not None:
            if verbose>=3: print('[gABiC] >Filter variables (nodes) on white_list..')
            white_list = [x.lower() for x in white_list]
            Iloc = np.isin(df.columns.str.lower(), white_list)
            df = df.loc[:, Iloc]

        # Exclude variables that are in black_list.
        if black_list is not None:
            if verbose>=3: print('[gABiC] >Filter variables (nodes) on black_list..')
            black_list = [x.lower() for x in black_list]
            Iloc = ~np.isin(df.columns.str.lower(), black_list)
            df = df.loc[:, Iloc]

        if (white_list is not None) or (black_list is not None):
            if verbose>=3: print('[gABiC] >Number of features after white/black listing: %d' %(df.shape[1]))
        if df.shape[1]<=1: raise Exception('[gABiC] >Error: [%d] variables are remaining. A minimum of 2 would be nice.' %(df.shape[1]))
    return df




# %% Constraint-based Structure Learning
def _pc_wrapper(df, significance_level=0.05, n_jobs=-1, verbose=3, variant="stable"):
    """Contraint-based BN structure learnging based on PC search algorithm.

    PC PDAG construction is only guaranteed to work under the assumption that the
    identified set of independencies is *faithful*, i.e. there exists a DAG that
    exactly corresponds to it. Spurious dependencies in the data set can cause
    the reported independencies to violate faithfulness. It can happen that the
    estimated PDAG does not have any faithful completions (i.e. edge orientations
    that do not introduce new v-structures). In that case a warning is issued.

    test_conditional_independence() returns a tripel (chi2, p_value, sufficient_data),
    consisting in the computed chi2 test statistic, the p_value of the test, and a heuristic
    flag that indicates if the sample size was sufficient.
    The p_value is the probability of observing the computed chi2 statistic (or an even higher chi2 value),
    given the null hypothesis that X and Y are independent given Zs.
    This can be used to make independence judgements, at a given level of significance.

    DAG (pattern) construction
    With a method for independence testing at hand, we can construct a DAG from the data set in three steps:
        1. Construct an undirected skeleton - `estimate_skeleton()`
        2. Orient compelled edges to obtain partially directed acyclid graph (PDAG; I-equivalence class of DAGs) - `skeleton_to_pdag()`
        3. Extend DAG pattern to a DAG by conservatively orienting the remaining edges in some way - `pdag_to_dag()`

        The first two steps form the so-called PC algorithm, see [2], page 550. PDAGs are `DirectedGraph`s, that may contain both-way edges, to indicate that the orientation for the edge is not determined.

    """
    if verbose>=4 and n_jobs>0: print('[gABiC] >n_jobs is not supported for [constraintsearch]')
    out = {}
    # Set structure-learning algorithm
    model = PC(df)

    #Building DAG skeleton
    N_max_Cvars = len(df.columns.values.tolist()) -2
    if (variant=="parallel") and (n_jobs>1):
       skel, seperating_sets = model.build_skeleton(max_cond_vars=N_max_Cvars,significance_level=significance_level, variant=variant, n_jobs=n_jobs)
    else:
       skel, seperating_sets = model.build_skeleton(max_cond_vars=N_max_Cvars,significance_level=significance_level)

    if verbose>=4: print("Undirected edges: ", skel.edges())
    pdag = model.skeleton_to_pdag(skel, seperating_sets)
    if verbose>=4: print("PDAG edges: ", pdag.edges())
    dag = pdag.to_dag()
    if verbose>=4: print("DAG edges: ", dag.edges())

    out['undirected'] = skel
    out['undirected_edges'] = skel.edges()
    out['pdag'] = pdag
    out['pdag_edges'] = pdag.edges()
    out['dag'] = dag
    out['dag_edges'] = dag.edges()

    # Search "estimate()" method provides a shorthand for the three steps above and directly returns a "BayesianNetwork"
    #if (variant=="dask-parallel") and (n_jobs>1):
    #   best_model = model.estimate(significance_level=significance_level, variant=variant, n_jobs=n_jobs)
    #else:
    best_model = model.estimate(significance_level=significance_level)

    out['model'] = best_model

    if verbose>=4: print(best_model.edges())
    return(out)



# %% Constraint-based Structure Learning
def _Kpc_wrapper(df, CI_test = 'hsic_gamma', significance_level=0.05, n_jobs=-1, verbose=3, variant="stable"):
    """Contraint-based BN structure learnging based on PC search algorithm.

    PC PDAG construction is only guaranteed to work under the assumption that the
    identified set of independencies is *faithful*, i.e. there exists a DAG that
    exactly corresponds to it. Spurious dependencies in the data set can cause
    the reported independencies to violate faithfulness. It can happen that the
    estimated PDAG does not have any faithful completions (i.e. edge orientations
    that do not introduce new v-structures). In that case a warning is issued.

    test_conditional_independence() returns a tripel (chi2, p_value, sufficient_data),
    consisting in the computed chi2 test statistic, the p_value of the test, and a heuristic
    flag that indicates if the sample size was sufficient.
    The p_value is the probability of observing the computed chi2 statistic (or an even higher chi2 value),
    given the null hypothesis that X and Y are independent given Zs.
    This can be used to make independence judgements, at a given level of significance.

    DAG (pattern) construction
    With a method for independence testing at hand, we can construct a DAG from the data set in three steps:
        1. Construct an undirected skeleton - `estimate_skeleton()`
        2. Orient compelled edges to obtain partially directed acyclid graph (PDAG; I-equivalence class of DAGs) - `skeleton_to_pdag()`
        3. Extend DAG pattern to a DAG by conservatively orienting the remaining edges in some way - `pdag_to_dag()`

        The first two steps form the so-called PC algorithm, see [2], page 550. PDAGs are `DirectedGraph`s, that may contain both-way edges, to indicate that the orientation for the edge is not determined.

    """
    if verbose>=4 and n_jobs>0: print('[gABiC] >n_jobs is not supported for [constraintsearch]')
    out = {}
    # Set structure-learning algorithm
    model = kPC(df)

    #Building DAG skeleton
    n_nodes = len(df.columns.values.tolist())
    if (variant=="parallel") and (n_jobs>1):
       skel, seperating_sets = model.build_skeleton(max_cond_vars=n_nodes,significance_level=significance_level, variant=variant, n_jobs=n_jobs)
    else:
       skel, seperating_sets = model.build_skeleton(max_cond_vars=n_nodes,significance_level=significance_level)

    if verbose>=4: print("Undirected edges: ", skel.edges())
    pdag = model.skeleton_to_pdag(skel, seperating_sets)
    if verbose>=4: print("PDAG edges: ", pdag.edges())
    dag = pdag.to_dag()
    if verbose>=4: print("DAG edges: ", dag.edges())

    out['undirected'] = skel
    out['undirected_edges'] = skel.edges()
    out['pdag'] = pdag
    out['pdag_edges'] = pdag.edges()
    out['dag'] = dag
    out['dag_edges'] = dag.edges()

    # Search "estimate()" method provides a shorthand for the three steps above and directly returns a "BayesianNetwork"
    #if (variant=="dask-parallel") and (n_jobs>1):
    #   best_model = model.estimate(significance_level=significance_level, variant=variant, n_jobs=n_jobs)
    #else:
    best_model = model.estimate(significance_level=significance_level)

    out['model'] = best_model

    if verbose>=4: print(best_model.edges())
    return(out)




# %%
def _is_independent(model, X, Y, Zs=[], significance_level=0.05):
    return model.test_conditional_independence(X, Y, Zs)[1] >= significance_level
