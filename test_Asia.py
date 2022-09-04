import pandas as pd
import numpy as np

import slearn as sl

df = sl.import_example('Asia')

#df.to_csv('Asia_10k_n.csv')
#ASIA_DAG = sl.import_DAG(filepath='asia')
#df = sl.sampling(ASIA_DAG, n=1000)

#printing to terminal the dataset
print(df.head(5))
# Structure learning
model_HC = sl.structure_learning.fit(df, methodtype='hc', verbose=4)
sl.plot(model_HC)

model_PC = sl.structure_learning.fit(df, methodtype='cs', verbose=4)
sl.plot(model_PC,interactive=True)




