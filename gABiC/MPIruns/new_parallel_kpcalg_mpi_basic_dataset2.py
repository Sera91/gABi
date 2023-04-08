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
from coreBN.estimators import kPC
from coreBN.base import PDAG
from dask_mpi import initialize
from dask.distributed import Client, as_completed
import dask.dataframe as dd
import params_basic as params




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
 

    BN = pKC(data=data_TNG)


    pDAG = BN.estimate(variant="dask_parallel", ci_test=iCI_test, random_seed= random_state, dask_client=dask_client, significance_level=0.05)

    ts2 = time.perf_counter() - ts
    print("Elapsed time [s] kernel-PC with test {} :  {:12.6f}".format(iCI_test,ts2))

    print(pDAG.edges())

    
    G_last = nx.DiGraph()

    G_last.add_edges_from( pDAG.edges())    

    nx.write_adjlist(G_last, file_adjlist)

    nx.write_edgelist(G_last,file_edgelist)

    dask_client.shutdown()


