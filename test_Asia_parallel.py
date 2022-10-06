import pandas as pd
import numpy as np

import slearn as sl

df = sl.import_example('Asia')

n_obs=len(df['asia'])

df.insert(0, 'index', np.arange(n_obs))

#printing to terminal the dataset
print(df.head(5))

arr_diff= (df['asia'] - df['tub']).to_numpy()
print(type(arr_diff))
#counting differences between asia and tub
print(len(arr_diff[np.where(arr_diff!= 0)])/n_obs)

#df.to_csv('Asia_10k_n.csv')
#ASIA_DAG = sl.import_DAG(filepath='asia')
#df = sl.sampling(ASIA_DAG, n=1000)



# Structure learning
#model_HC = sl.structure.learn(df, method='hc', verbose=4)
#sl.plot(model_HC)

model_PC = sl.structure.learn(df, method='pc', n_jobs=4, verbose=4, variant="parallel")
sl.plot(model_PC)#,interactive=True)




