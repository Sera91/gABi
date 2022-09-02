"""Parameter learning.

Description
-----------
Parameter learning is the task to estimate the values of the conditional probability distributions (CPDs).
To make sense of the given data, we can start by counting how often each state of the variable occurs.
If the variable is dependent on the parents, the counts are done conditionally on the parents states,
i.e. for seperately for each parent configuration

Currently, the library supports parameter learning for *discrete* nodes:
    * Maximum Likelihood Estimation
    * Bayesian Estimation
"""

from coreBN.estimators import BayesianEstimator
# from pgmpy.estimators import MaximumLikelihoodEstimator
import slearn.base as slearn
import warnings
warnings.filterwarnings("ignore")


# %% Sampling from model
def fit(model, df, methodtype='bayes', scoretype='bdeu', smooth=None, n_jobs=-1, verbose=3):
    """Learn the parameters given the DAG and data.

    Description
    -----------
    Maximum Likelihood Estimation
        A natural estimate for the CPDs is to simply use the *relative frequencies*,
        with which the variable states have occured. We observed x cloudy` among a total of `all clouds`,
        so we might guess that about `50%` of `cloudy` are `sprinkler or so.
        According to MLE, we should fill the CPDs in such a way, that P(data|model) is maximal.
        This is achieved when using the relative frequencies.

        While very straightforward, the ML estimator has the problem of *overfitting* to the data.
        If the observed data is not representative for the underlying distribution, ML estimations will be extremly far off.
        When estimating parameters for Bayesian networks, lack of data is a frequent problem.
        Even if the total sample size is very large, the fact that state counts are done conditionally
        for each parents configuration causes immense fragmentation.
        If a variable has 3 parents that can each take 10 states, then state counts will
        be done seperately for `10^3 = 1000` parents configurations.
        This makes MLE very fragile and unstable for learning Bayesian Network parameters.
        A way to mitigate MLE's overfitting is *Bayesian Parameter Estimation*.

    Bayesian Parameter Estimation
        The Bayesian Parameter Estimator starts with already existing prior CPDs,
        that express our beliefs about the variables *before* the data was observed.
        Those "priors" are then updated, using the state counts from the observed data.

        One can think of the priors as consisting in *pseudo state counts*, that are added
        to the actual counts before normalization. Unless one wants to encode specific beliefs
        about the distributions of the variables, one commonly chooses uniform priors,
        i.e. ones that deem all states equiprobable.

        A very simple prior is the so-called *K2* prior, which simply adds `1` to the count of every single state.
        A somewhat more sensible choice of prior is *BDeu* (Bayesian Dirichlet equivalent uniform prior).
        For BDeu we need to specify an *equivalent sample size* `N` and then the pseudo-counts are
        the equivalent of having observed `N` uniform samples of each variable (and each parent configuration).

    Parameters
    ----------
    model : dict
        Contains a model object with a key 'adjmat' (adjacency matrix).
    df : pd.DataFrame() #Pandas DataFrame containing the data.
    methodtype : str, (default: 'bayes')
        Strategy for parameter learning.
            * 'ml', 'maximumlikelihood': Learning CPDs using Maximum Likelihood Estimators.
            * 'bayes': Bayesian Parameter Estimation.
    scoretype : str, (default : 'bic')
        Scoring function for the search spaces.
            * 'bdeu'
            * 'dirichlet'
            * 'k2'
    smooth : float (default: None)
        The smoothing value (α) for Bayesian parameter estimation. Should be Nonnegative.
    verbose: guiding the printing process on the screen. 
             0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE
             The default is 3.

    Returns
    -------
    dict with model.

    Examples
    --------
    >>> import slearn as bn
    >>>
    >>> df = bn.import_example()
    >>> model = bn.import_DAG('sprinkler', CPD=False)
    >>>
    >>> # Parameter learning
    >>> model_update = bn.parameter_learning.fit(model, df)
    >>> bn.plot(model_update)
    >>>
    >>> # LOAD BIF FILE
    >>> model = bn.import_DAG('alarm')
    >>> df = bn.sampling(model, n=1000)
    >>> model_update = bn.parameter_learning.fit(model, df)
    >>> G = bn.plot(model_update)

    """
    config = {}
    config['verbose'] = verbose
    config['method'] = methodtype
    config['n_jobs'] = n_jobs
    adjmat = model['adjmat']

    if (scoretype=='dirichlet') and (smooth is None):
        raise Exception('[slearn] >dirichlet requires "smooth" to be not None')

    # Check whether all labels in the adjacency matrix are included from the dataframe
    # adjmat, model = _check_adjmat(model, df)
    df = slearn._filter_df(adjmat, df, verbose=config['verbose'])

    if config['verbose']>=3: print('[slearn] >Parameter learning> Computing parameters using [%s]' %(config['method']))
    # Extract model
    if isinstance(model, dict):
        model = model['model']

    # Convert to BayesianNetwork
    if 'BayesianNetwork' not in str(type(model)):
        model = slearn.to_bayesiannetwork(adjmat, verbose=config['verbose'])

    # Learning CPDs using Maximum Likelihood Estimators
    if config['method']=='ml' or config['method']=='maximumlikelihood':
        model.fit(df, estimator=None)  # estimator as None makes it maximum likelihood estimator according pgmpy docs.
        for cpd in model.get_cpds():
            if config['verbose']>=3: print("[slearn] >CPD of {variable}:".format(variable=cpd.variable))
            if config['verbose']>=3: print(cpd)

    #  Learning CPDs using Bayesian Parameter Estimation
    if config['method']=='bayes':
        model.fit(df, estimator=BayesianEstimator, prior_type=scoretype, equivalent_sample_size=1000, pseudo_counts=smooth, n_jobs=config['n_jobs'])
        # model.fit(df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=1000, pseudo_counts=smooth)
        for cpd in model.get_cpds():
            if config['verbose']>=3: print("[slearn] >CPD of {variable}:".format(variable=cpd.variable))
            if config['verbose']>=3: print(cpd)

    out = {}
    out['model'] = model
    out['adjmat'] = adjmat
    out['config'] = config
    out['model_edges'] = list(model.edges())

    return(out)
