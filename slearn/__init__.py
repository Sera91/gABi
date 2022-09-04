from slearn.base import (
    to_bayesiannetwork,
    make_DAG,
    print_CPD,
    import_DAG,
    import_example,
    sampling,
    to_undirected,
    compare_networks,
    plot,
    adjmat2vec,
    adjmat2dict,
    vec2adjmat,
    dag2adjmat,
    df2onehot,
    topological_sort,
    predict,
    query2df,
    vec2df,
    get_node_properties,
    get_edge_properties,
    _filter_df,
    _bif2bayesian,
    independence_test,
    save,
    load,
)

# Import function in new level
import slearn.Learn_struct as structure_learning
import slearn.Learn_par as parameter_learning
import slearn.Inf as inference
import slearn.Netk as network
import slearn.CM as confmatrix
import coreBN
from packaging import version



# Version check
import matplotlib
if not version.parse(matplotlib.__version__) >= version.parse("3.3.4"):
    raise ImportError('[bnlearn] >Error: Matplotlib version should be >= v3.3.4.\nTry to: pip install -U matplotlib')

import networkx as nx
if not version.parse(nx.__version__) >= version.parse("2.7.1"):
    raise ImportError('[bnlearn] >Error: networkx version should be > 2.7.1.\nTry to: pip install -U networkx')


# module level doc-string
__doc__ = """
SLEARN - Slearn is a Python package for learning the graphical structure of Bayesian networks, estimate their parameters, perform inference, and comparing networks.
================================================================================================================================================================================

Description
-----------
* Learning a Bayesian network can be split into two problems:
    * Structure learning: Given a set of data samples, estimate a DAG that captures the dependencies between the variables.
    * Parameter learning: Given a set of data samples and a DAG that captures the dependencies between the variables.
    

* Structure learning algorithms can be divided into two classes:
        * Score-based structure learning (BIC/BDeu/K2 score; exhaustive search, hill climb/tabu search)
        * Constraint-based structure learning (here we implement only PC-stable)
     


Example
-------
>>> # Import library
>>> import Slearn as sl
>>> model = sl.import_DAG('sprinkler')
>>> # Print CPDs
>>> sl.print_CPD(model)
>>> # Plot DAG
>>> sl.plot(model)
>>>
>>> # Sampling using DAG and CPDs
>>> df = sl.sampling(model)
>>>
>>> # Do the inference
>>> q1 = sl.inference.fit(model, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
>>> q2 = sl.inference.fit(model, variables=['Wet_Grass','Rain'], evidence={'Sprinkler':1})
>>>
>>> # Structure learning
>>> model_sl = sl.structure_learning.fit(df)
>>> # Compute edge strength using chi-square independence test
>>> model_sl = sl.independence_test(model_sl, df)
>>> # Plot DAG
>>> sl.plot(model_sl)
>>>
>>> # Parameter learning
>>> model_pl = sl.parameter_learning.fit(model_sl, df)
>>> # Compute edge strength using chi-square independence test
>>> model_pl = sl.independence_test(model_pl, df)
>>> # Plot DAG
>>> sl.plot(model_pl)
>>>
>>> # Compare networks
>>> scores, adjmat = sl.compare_networks(model_sl, model)

References
----------
* https://bnlearn.readthedocs.io
* https://towardsdatascience.com/a-step-by-step-guide-in-detecting-causal-relationships-using-bayesian-structure-learning-in-python-c20c6b31cee5
* https://towardsdatascience.com/a-step-by-step-guide-in-designing-knowledge-driven-models-using-bayesian-theorem-7433f6fd64be

"""
