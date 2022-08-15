import pandas as pd
import numpy as np

import slearn as sl

df = sl.import_example('Asia')

#printing to terminal the dataset
df.head(5)
# Structure learning
model_HC = sl.structure_learning.fit(df, methodtype='hc', verbose=4)
sl.plot(model_HC)

model_PC = sl.structure_learning.fit(df, methodtype='cs', verbose=4, n_jobs=1)
sl.plot(model_PC)
