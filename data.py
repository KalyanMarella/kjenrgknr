import pandas as pd
import numpy as np

df=pd.DataFrame()
df["X1"]=np.random.randn((100))
df["X2"]=np.log(abs(np.random.randn(100)))
#print(df)
df["target"]=df["X1"]+2*df["X2"]

df.to_csv("data/data.csv",index=False)
