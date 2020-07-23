import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('predictedNifty2010_2020.csv',sep='\t',index_col='Date')

plot1 = df1[['Nifty Actual','Nifty predicted']].plot(figsize=(15,15))

plot1.get_figure().savefig('predictedNifty2010_2020.png')

df2 = pd.read_csv('predictedNifty2015_2020.csv',sep='\t',index_col='Date')

plot2 = df2[['Nifty Actual','Nifty predicted']].plot(figsize=(15,15))

plot2.get_figure().savefig('predictedNifty2015_2020.png')