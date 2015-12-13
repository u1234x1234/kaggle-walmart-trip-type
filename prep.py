import pandas as pd


x = pd.read_csv('./test.csv')
print (x)

x.to_csv('prep_test.csv', sep='_', index=None)
 
