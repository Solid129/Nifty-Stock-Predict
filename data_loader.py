from nsepy import get_history, get_index_pe_history
from datetime import date, timedelta, datetime
from calendar import monthrange
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt
import numpy as np

def findExpiryDate(l):
    day = l.isoweekday()
    if(day == 4):
        return l
    if(day > 4):
        return l-timedelta(days=(day-4))
    if(day < 4):
        return l-timedelta(days=(7+day-4))


def nifty_getHistory(s, l):
    return get_history(symbol="NIFTY",
                       start=s,
                       end=l,
                       index=True,
                       futures=True,
                       expiry_date=l)

def nifty50Data(start,end):
    months = 13
    s = start
    if start.year==end.year:
        months = end.month+1
    f = open('nifty{0}_{1}.csv'.format(start.year,end.year),'w')
    for year in range(start.year,end.year+1):
        for month in range(1,months):
            l = findExpiryDate(date(year,month,1)+timedelta(days=monthrange(year,month)[1]-1))
            nifty = nifty_getHistory(s,l)
            #print(nifty.dtypes)
            s = l + timedelta(days=1)
            if(start.year==year and month==1):
                nifty.to_csv(f,sep='\t',columns=['Close','Open Interest','Change in OI'])
            else:
                nifty.to_csv(f,header=False,sep='\t',columns=['Close','Open Interest','Change in OI'])
    f.close()
    nifty = pd.read_csv('nifty{0}_{1}.csv'.format(start.year,end.year),index_col='Date',sep='\t')
    nifty['SMA'] = ta.SMA(nifty['Close'],50)
    nifty['EMA'] = ta.EMA(nifty['Close'], timeperiod = 5)
    nifty['upper_band'], nifty['middle_band'], nifty['lower_band'] = ta.BBANDS(nifty['Close'], timeperiod =20)
    nifty['Percentage Change'] = nifty['Close'].diff()*100/nifty['Close']
    nifty.to_csv('nifty{0}_{1}.csv'.format(start.year,end.year),sep='\t')
    

        
start = date(2010, 1, 1)
end = date(2020, 7, 22)


nifty50Data(start,end)

vix = get_history(symbol="INDIAVIX",
                  start=start,
                  end=end,
                  index=True)

nifty_pe = get_index_pe_history(symbol="NIFTY",
                                start=start,
                                end=end)

vix.to_csv('vix{0}_{1}.csv'.format(start.year,end.year), sep='\t',columns=['Close'])
nifty_pe.to_csv('niftyPE{0}_{1}.csv'.format(start.year,end.year), sep='\t',columns=['P/E'])

dowJones = pd.read_csv('Dow Jones{0}_{1}.csv'.format(2010,end.year),thousands=',')
dowJones['Date'] = dowJones['Date'].map(lambda x: datetime.strptime(x, "%b %d, %Y"))
dowJones.to_csv('DowJones{0}_{1}.csv'.format(start.year,end.year),sep='\t',columns=['Date','Price'],index=False)

#Merge data altogether
a = pd.read_csv('DowJones{0}_{1}.csv'.format(start.year,end.year), sep='\t', index_col='Date')
b = pd.read_csv('niftyPE{0}_{1}.csv'.format(start.year,end.year), sep='\t', index_col='Date')
c = pd.read_csv('nifty{0}_{1}.csv'.format(start.year,end.year), sep='\t', index_col='Date')
d = pd.read_csv('vix{0}_{1}.csv'.format(start.year,end.year), sep='\t', index_col='Date')

# as vix as some NaN values
d.dropna(inplace=True)

#renaming columns
a = a.rename(columns={'Price':'Dow Jones'})
d = d.rename(columns={'Close':'VIX'})
e = pd.concat([a, b, c, d], axis=1, sort=False)
e.index.name = 'Date'
e.to_csv('nifty_data{0}_{1}.csv'.format(start.year,end.year), sep='\t')
