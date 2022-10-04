"""Structure learning. Given a set of data samples, estimate a DAG that captures the dependencies between the variables."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from coreBN.estimators import BDeuScore, K2Score, BicScore
from coreBN.estimators import ExhaustiveSearch, HillClimbSearch

from coreBN.estimators.PC import PC as PC


import slearn.base as slearn


# %% Structure Learning
def fit(df, methodtype='hc', scoretype='bic', black_list=None, white_list=None, bw_list_method=None, max_indegree=None, tabu_length=100, epsilon=1e-4, max_iter=1e6, root_node=None, class_node=None, fixed_edges=None, return_all_dags=False, n_jobs=-1, verbose=3):
    """Structure learning fit model.

    Description
    -----------
    Search strategies for structure learning
    The search space of DAGs is super-exponential in the number of variables and the above scoring functions allow for local maxima.

    To learn model structure (a DAG) from a data set, there are 2 broad techniques:
        1. Score-based structure learning (BIC/BDeu/K2 score; exhaustive search, hill climb/tabu search):
            a) exhaustivesearch
            b) hillclimbsearch
            
        2. Constraint-based structure learning:
            a. PC (based in indep tests)
            b. IAMB
        

    
    The score-based Structure Learning approach has two building blocks:
       - a scoring function sD:->R that maps models to a numerical score, based on how well they fit to a given data set D.
       - a search strategy to traverse the search space of possible models M and select a model with optimal score.
    Commonly used scoring functions to measure the fit between model and data are Bayesian Dirichlet scores such as BDeu or K2 and the Bayesian Information Criterion (BIC, also called MDL).
    BDeu is dependent on an equivalent sample size.

    Parameters
    ----------
    df : pd.DataFrame()
        Input dataframe.
    methodtype : str, (default : 'hc')
        String Search strategy for structure_learning.
        'hc' or 'hillclimbsearch' (default)
        'ex' or 'exhaustivesearch'
        'cs' or 'constraintsearch'
        
    scoretype : str, (default : 'bic')
        Scoring function for the search spaces.
        'bic', 'k2', 'bdeu'
    black_list : List or None, (default : None)
        List of edges are black listed.
        In case of filtering on nodes, the nodes black listed nodes are removed from the dataframe. The resulting model will not contain any nodes that are in black_list.
    white_list : List or None, (default : None)
        List of edges are white listed.
        In case of filtering on nodes, the search is limited to those edges. The resulting model will then only contain nodes that are in white_list.
        Works only in case of methodtype='hc' See also paramter: `bw_list_method`
    bw_list_method : list of str or tuple, (default : None)
        A list of edges can be passed as `black_list` or `white_list` to exclude or to limit the search.
            * 'edges' : [('A', 'B'), ('C','D'), (...)] 
            * 'nodes' : ['A', 'B', ...] Filter the dataframe based on the nodes for `black_list` or `white_list`. Filtering can be done for every methodtype/scoretype.
    max_indegree : int, (default : None)
        If provided and unequal None, the procedure only searches among models where all nodes have at most max_indegree parents. (only in case of methodtype='hc')
    epsilon: float (default: 1e-4)
        Defines the exit condition. If the improvement in score is less than `epsilon`, the learned model is returned. (only in case of methodtype='hc')
    max_iter: int (default: 1e6)
        The maximum number of iterations allowed. Returns the learned model when the number of iterations is greater than `max_iter`. (only in case of methodtype='hc')
    root_node: String. (only in case of chow-liu, Tree-augmented Naive Bayes (TAN))
        The root node for treeSearch based methods.
    class_node: String
        The class node is required for Tree-augmented Naive Bayes (TAN)
    fixed_edges: iterable, Only in case of HillClimbSearch.
        A list of edges that will always be there in the final learned model. The algorithm will add these edges at the start of the algorithm and will never change it.
    return_all_dags : Bool, (default: False)
        Return all possible DAGs. Only in case methodtype='exhaustivesearch'
    verbose : int, (default : 3)
        0: None, 1: Error,  2: Warning, 3: Info (default), 4: Debug, 5: Trace

    Returns
    -------
    dict with model.

    Examples
    --------
    >>> # Import slearn
    >>> import slearn as sl
    >>> # Load DAG
    >>> model = sl.import_DAG('asia')
    >>> # plot ground truth
    >>> G = sl.plot(model)
    >>> # Sampling
    >>> df = sl.sampling(model, n=10000)
    >>> # Structure learning of sampled dataset
    >>> model_sl = sl.structure_learning.fit(df, methodtype='hc', scoretype='bic')
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
    config = {'method': methodtype, 'scoring': scoretype, 'black_list': black_list, 'white_list': white_list, 'bw_list_method': bw_list_method, 'max_indegree': max_indegree, 'tabu_length': tabu_length, 'epsilon': epsilon, 'max_iter': max_iter, 'root_node': root_node, 'class_node': class_node, 'fixed_edges': fixed_edges, 'return_all_dags': return_all_dags, 'n_jobs': n_jobs, 'verbose': verbose}
    # Make some checks
    config = _make_checks(df, config, verbose=verbose)
    # Make sure columns are of type string
    df.columns = df.columns.astype(str)
    # Filter on white_list and black_list
    df = _white_black_list_filter(df, white_list, black_list, bw_list_method=config['bw_list_method'], verbose=verbose)
    # Lets go!
    if config['verbose']>=3: print('[slearn] >Computing best DAG using [%s]' %(config['method']))

    

    # ExhaustiveSearch can be used to compute the score for every DAG and returns the best-scoring one:
    if config['method']=='ex' or config['method']=='exhaustivesearch':
        out = _exhaustivesearch(df,
                                scoretype=config['scoring'],
                                return_all_dags=config['return_all_dags'],
                                n_jobs=config['n_jobs'],
                                verbose=config['verbose'])

    # HillClimbSearch
    if config['method']=='hc' or config['method']=='hillclimbsearch':
        out = _hillclimbsearch(df,
                               scoretype=config['scoring'],
                               black_list=config['black_list'],
                               white_list=config['white_list'],
                               max_indegree=config['max_indegree'],
                               tabu_length=config['tabu_length'],
                               bw_list_method=bw_list_method,
                               epsilon=config['epsilon'],
                               max_iter=config['max_iter'],
                               fixed_edges=config['fixed_edges'],
                               n_jobs=config['n_jobs'],
                               verbose=config['verbose'],
                               )

    # Constraint-based Structure Learning
    if config['method']=='cs' or config['method']=='constraintsearch':
        """Constraint-based Structure Learning
        
        Construct DAG (pattern) according to identified independencies between vars, based on (Conditional) Independence Tests.
        """
        out = _constraintsearch(df, n_jobs=config['n_jobs'], verbose=config['verbose'])

    
    # 
    out['model_edges'] = list(out['model'].edges())
    out['adjmat'] = slearn.dag2adjmat(out['model'])
    out['config'] = config

    # return
    return(out)


# %% Make Checks
def _make_checks(df, config, verbose=3):
    assert isinstance(pd.DataFrame(), type(df)), 'df must be of type pd.DataFrame()'
    if not np.isin(config['scoring'], ['bic', 'k2', 'bdeu']): raise Exception('"scoretype=%s" is invalid.' %(config['scoring']))
    if not np.isin(config['method'], ['hc', 'ex', 'cs', 'exhaustivesearch', 'hillclimbsearch', 'constraintsearch']): raise Exception('"methodtype=%s" is invalid.' %(config['method']))

    if isinstance(config['white_list'], str):
        config['white_list'] = [config['white_list']]
    if isinstance(config['black_list'], str):
        config['black_list'] = [config['black_list']]

    if (config['white_list'] is not None) and len(config['white_list'])==0:
        config['white_list'] = None
    if (config['black_list'] is not None) and len(config['black_list'])==0:
        config['black_list'] = None

    if (config['method']!='hc') and (config['bw_list_method']=='edges'): raise Exception('[slearn] >The "bw_list_method=%s" does not work with "methodtype=%s"' %(config['bw_list_method'], config['method']))
   
    if config['fixed_edges'] is None:
        config['fixed_edges']=set()

    # Show warnings
    if (config['bw_list_method'] is None) and ((config['black_list'] is not None) or (config['white_list'] is not None)):
        raise Exception('[slearn] >Error: The use of black_list or white_list requires setting bw_list_method.')
    if df.shape[1]>10 and df.shape[1]<15:
        if verbose>=2: print('[slearn] >Warning: Computing DAG with %d nodes can take a very long time!' %(df.shape[1]))
    if (config['max_indegree'] is not None) and config['method']!='hc':
        if verbose>=2: print('[slearn] >Warning: max_indegree only works in case of methodtype="hc"')
    if (config['class_node'] is not None) and config['method']!='tan':
        if verbose>=2: print('[slearn] >Warning: max_indegree only works in case of methodtype="tan"')

    return config





# %% white_list and black_list
def _white_black_list_filter(df, white_list, black_list, bw_list_method='edges', verbose=3):
    if bw_list_method=='nodes':
        # Keep only variables that are in white_list.
        if white_list is not None:
            if verbose>=3: print('[slearn] >Filter variables (nodes) on white_list..')
            white_list = [x.lower() for x in white_list]
            Iloc = np.isin(df.columns.str.lower(), white_list)
            df = df.loc[:, Iloc]

        # Exclude variables that are in black_list.
        if black_list is not None:
            if verbose>=3: print('[slearn] >Filter variables (nodes) on black_list..')
            black_list = [x.lower() for x in black_list]
            Iloc = ~np.isin(df.columns.str.lower(), black_list)
            df = df.loc[:, Iloc]

        if (white_list is not None) or (black_list is not None):
            if verbose>=3: print('[slearn] >Number of features after white/black listing: %d' %(df.shape[1]))
        if df.shape[1]<=1: raise Exception('[slearn] >Error: [%d] variables are remaining. A minimum of 2 would be nice.' %(df.shape[1]))
    return df




# %% Constraint-based Structure Learning
def _constraintsearch(df, significance_level=0.05, n_jobs=-1, verbose=3):
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
    if verbose>=4 and n_jobs>0: print('[slearn] >n_jobs is not supported for [constraintsearch]')
    out = {}
    # Set search algorithm
    model = PC(df)

    # Estimate using chi2
    n_nodes = len(df.columns.values.tolist())
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

    # Search using "estimate()" method provides a shorthand for the three steps above and directly returns a "BayesianNetwork"
    best_model = model.estimate(significance_level=significance_level)
    out['model'] = best_model

    if verbose>=4: print(best_model.edges())
    return(out)


# %% hillclimbsearch
def _hillclimbsearch(df, scoretype='bic', black_list=None, white_list=None, max_indegree=None, tabu_length=100, epsilon=1e-4, max_iter=1e6, bw_list_method='edges', fixed_edges=set(), n_jobs=-1, verbose=3):
    """Heuristic hill climb searches for DAGs, to learn network structure from data. `estimate` attempts to find a model with optimal score.

    Description
    -----------
    Performs local hill climb search to estimates the `DAG` structure
    that has optimal score, according to the scoring method supplied in the constructor.
    Starts at model `start` and proceeds by step-by-step network modifications
    until a local maximum is reached. Only estimates network structure, no parametrization.

    Once more nodes are involved, one needs to switch to heuristic search.
    HillClimbSearch implements a greedy local search that starts from the DAG
    "start" (default: disconnected DAG) and proceeds by iteratively performing
    single-edge manipulations that maximally increase the score.
    The search terminates once a local maximum is found.

    For details on scoring see Koller & Friedman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
    If a number `max_indegree` is provided, only modifications that keep the number
    of parents for each node below `max_indegree` are considered. A list of
    edges can optionally be passed as `black_list` or `white_list` to exclude those
    edges or to limit the search.

    """
    if verbose>=4 and n_jobs>0: print('[slearn] >n_jobs is not supported for [hillclimbsearch]')
    out={}
    # Set scoring type
    scoring_method = _SetScoringType(df, scoretype, verbose=verbose)
    # Set search algorithm
    model = HillClimbSearch(df)

    # Compute best DAG
    if bw_list_method=='edges':
        if (black_list is not None) or (white_list is not None):
            if verbose>=3: print('[slearn] >Filter edges based on black_list/white_list')
        # best_model = model.estimate()
        best_model = model.estimate(scoring_method=scoring_method, max_indegree=max_indegree, tabu_length=tabu_length, epsilon=epsilon, max_iter=max_iter, black_list=black_list, white_list=white_list, fixed_edges=fixed_edges, show_progress=False)
    else:
        # At this point, variables are readily filtered based on bw_list_method or not (if nothing defined).
        best_model = model.estimate(scoring_method=scoring_method, max_indegree=max_indegree, tabu_length=tabu_length, epsilon=epsilon, max_iter=max_iter, fixed_edges=fixed_edges, show_progress=False)

    # Store
    out['model']=best_model
    # Return
    return(out)


# %% ExhaustiveSearch
def _exhaustivesearch(df, scoretype='bic', return_all_dags=False, n_jobs=-1, verbose=3):
    """Exhaustivesearch.

    Description
    ------------
    The first property makes exhaustive search intractable for all but very
    small networks, the second prohibits efficient local optimization
    algorithms to always find the optimal structure. Thus, identifiying the
    ideal structure is often not tractable. Despite these bad news, heuristic
    search strategies often yields good results if only few nodes are involved
    (read: less than 5).

    Parameters
    ----------
    df : pandas DataFrame object
        A DataFrame object with column names same as the variable names of network.
    scoretype : str, (default : 'bic')
        Scoring function for the search spaces.
        'bic', 'k2', 'bdeu'
    return_all_dags : Bool, (default: False)
        Return all possible DAGs.
    verbose : int, (default : 3)
        0:None, 1:Error, 2:Warning, 3:Info (default), 4:Debug, 5:Trace

    Returns
    -------
    None.

    """
    if df.shape[1]>15 and verbose>=3:
        print('[slearn] >Warning: Structure learning with more then 15 nodes is computationally not feasable with exhaustivesearch. Use hillclimbsearch or constraintsearch instead!')  # noqa
    if verbose>=4 and n_jobs>0: print('[slearn] >n_jobs is not supported for [exhaustivesearch]')

    out={}
    # Set scoring type
    scoring_method = _SetScoringType(df, scoretype, verbose=verbose)
    # Exhaustive search across all dags
    model = ExhaustiveSearch(df, scoring_method=scoring_method)
    # Compute best DAG
    best_model = model.estimate()
    # Store
    out['model']=best_model

    # Compute all possible DAGs
    if return_all_dags:
        out['scores']=[]
        out['dag']=[]
        # print("\nAll DAGs by score:")
        for [score, dag] in reversed(model.all_scores()):
            out['scores'].append(score)
            out['dag'].append(dag)
            # print(score, dag.edges())

        plt.plot(out['scores'])
        plt.show()

    return(out)


# %% Set scoring type
def _SetScoringType(df, scoretype, verbose=3):
    if verbose>=3: print('[slearn] >Set scoring type at [%s]' %(scoretype))

    if scoretype=='bic':
        scoring_method = BicScore(df)
    elif scoretype=='k2':
        scoring_method = K2Score(df)
    elif scoretype=='bdeu':
        scoring_method = BDeuScore(df, equivalent_sample_size=5)

    return(scoring_method)


# %%
def _is_independent(model, X, Y, Zs=[], significance_level=0.05):
    return model.test_conditional_independence(X, Y, Zs)[1] >= significance_level
