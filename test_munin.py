import pandas as pd
import numpy as np
import time

import slearn as sl

df = pd.read_csv("munin_iamb20k.csv")

nodes= (df.columns).to_list()
n_obs=df.shape[0] #number of rows in the dataframe

df_sel=df.iloc[:, 0:30]

for name in (df_sel.columns).to_list():
    df_sel[name] = pd.Categorical(df_sel[name])
    df_sel[name] = df_sel[name].cat.codes


#printing to terminal the dataset
print(df_sel.head(5))

#print summary statistics

#print(df.describe())




# Structure learning
#model_HC = sl.structure.learn(df, method='hc', verbose=4)
#sl.plot(model_HC)
t1=time.time()
model_PC = sl.structure.learn(df_sel, method='pc', n_jobs=4, verbose=4, variant="parallel")
t2=time.time() - t1

print("seconds running PC in SLEARN:", t2)

#sl.plot(model_PC)#,interactive=True)




