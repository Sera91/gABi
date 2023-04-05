import pandas as pd
import numpy as np
#import timemory
import time
import argparse
import gABiC as gb
#from timemory.component import PapiVector



if __name__ == "__main__":

    ts = time.perf_counter()

    df = gb.import_example('Asia')
    
    ts = time.perf_counter() - ts
    print("Elapsed time reading dataset (sec) :  {:12.6f}".format(ts))

    n_obs=len(df['asia'])

    df.insert(0, 'index', np.arange(n_obs))

    #hwc = PapiVector("MY_REGION_OF_INTEREST")  # label when printing
    #hwc.start()

    t2= time.perf_counter() 

    #printing to terminal the dataset
    #print(df.head(5))
    #arr_diff= (df['asia'] - df['tub']).to_numpy()
    #print(type(arr_diff))
    #counting differences between asia and tub
    #print(len(arr_diff[np.where(arr_diff!= 0)])/n_obs)

    #df.to_csv('Asia_10k_n.csv')
    #ASIA_DAG = sl.import_DAG(filepath='asia')
    #df = sl.sampling(ASIA_DAG, n=1000)

    # Structure learning
    #model_HC = sl.structure.learn(df, method='hc', verbose=4)
    #sl.plot(model_HC)

    model_PC = gb.structure.learn(df, method='pc', n_jobs=4, verbose=4, variant="parallel")
    #sl.plot(model_PC)#,interactive=True)
    ts2 = time.perf_counter() - t2
    print("Elapsed time PC (sec) :  {:12.6f}".format(ts2))




