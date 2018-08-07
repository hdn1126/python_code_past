
# coding: utf-8

# In[3]:


## times series plot of churn rate 
import pandas as pd
import numpy as np
from datetime import datetime
import csv
from pandas import Series, DataFrame, Panel
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)
df['date'] = pd.to_datetime(df['date'])

plt.plot(df['date'], df['churn'])
plt.title('yearly timeseries') 
plt.xlabel('Time') 
plt.ylabel('Churn rate(%)')

plt.show()


# In[4]:


## Is the data normal? : No
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)
from scipy.stats import shapiro


stat, p = shapiro(df['churn'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')


# In[5]:


## churn rates histogram
import pandas as pd
from datetime import datetime
import csv
from pandas import Series, DataFrame, Panel
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)


plt.hist(df['churn'], bins = 50 )
plt.title('distribution of churn rates') 
plt.xlabel('Churn rates') 
plt.ylabel('number')

mean = df['churn'].mean()
std = df['churn'].std()
upper_bound = mean+3*std

plt.axvline(x=upper_bound, color='r', linestyle='-')
plt.show()
print(upper_bound)


# In[6]:


## churn rates histogram excluding extreme values
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)

noextreme= df['churn'][np.abs(df['churn']-df['churn'].mean())<=(4*df['churn'].std())]

mean = noextreme.mean()
std =  noextreme.std()
upper_bound = mean+2*std
plt.axvline(x=upper_bound, color='r', linestyle='-')

countextreme = len(noextreme[noextreme>upper_bound])

plt.hist(noextreme, bins = 50 )
print(upper_bound)
print(countextreme)


# In[7]:


## outliers with boxplots 
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)


outlierfree= df['churn'][np.abs(df['churn']-df['churn'].mean())<=(4*df['churn'].std())]


# basic plot
plt.boxplot(outlierfree)
# box plots are good for symmetric and continuous distributions
plt.title('box plots') 


# In[8]:


#normalizing the data 
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)


replaced = df['churn'].replace(0, 0.00001)
df['y'], lam = boxcox(replaced)
plt.hist(df['y'], bins = 50 )

mean = df['y'].mean()
std =  df['y'].std()
upper_bound = mean+3*std
lower_bound = mean-3*std
plt.axvline(x=upper_bound, color='r', linestyle='-')
plt.axvline(x=lower_bound, color='b', linestyle='-')

plt.hist(df['y'], bins = 50 )
print(upper_bound)


# In[9]:


#boxplot of normalized data 
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)


replaced = df['churn'].replace(0, 0.00001)
df['y'], lam = boxcox(replaced)
plt.boxplot(df['y'])


# In[10]:


##  normalized without outliers 
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)


noextreme= df['churn'][np.abs(df['churn']-df['churn'].mean())<=(3*df['churn'].std())]
#noextreme= noextreme.remove('NaN')
replaced = noextreme.replace(0, 0.00001)
noextreme['y'], lam = boxcox(replaced)
plt.hist(noextreme['y'], bins = 50 )


mean = noextreme['y'].mean()
std =  noextreme['y'].std()
upper_bound = mean+2*std
lower_bound = mean-2*std
plt.axvline(x=upper_bound, color='r', linestyle='-')
plt.axvline(x=lower_bound, color='b', linestyle='-')

plt.hist(noextreme['y'], bins = 50 )
print(upper_bound)


# In[11]:


## normalized plot without the zeros 
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)

## delete the big extremes and zeros
noextreme= df['churn'][np.abs(df['churn']-df['churn'].mean())<=(3*df['churn'].std())]
noextreme= noextreme[noextreme>0]

#normalize the data using boxcox
noextreme['y'], lam = boxcox(noextreme)
#plt.hist(noextreme['y'], bins = 50 )

# mean / std
mean = noextreme['y'].mean()
std =  noextreme['y'].std()
upper_bound = mean+3*std
lower_bound = mean-2*std
outlier_count = len(noextreme['y'][noextreme['y'] > upper_bound])

#plot
plt.axvline(x=upper_bound, color='r', linestyle='-')
plt.axvline(x=lower_bound, color='b', linestyle='-')
plt.hist(noextreme['y'], bins = 50 )

print(upper_bound)
print(outlier_count)
print("Note that we have 0 outliers with 3*std and 22 with 2*std")


# In[12]:


## locating the outliers
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)

## delete the big extremes and zeros
noextreme= df['churn'][np.abs(df['churn']-df['churn'].mean())<=(3*df['churn'].std())]
countextreme = len(df['churn'][np.abs(df['churn']-df['churn'].mean())>(3*df['churn'].std())])
noextreme= noextreme[noextreme>0]

#normalize the data using boxcox
noextreme['y'], lam = boxcox(noextreme)

# mean / std for plotting purpose
mean = noextreme['y'].mean()
std =  noextreme['y'].std()
upper_bound = mean+2*std
lower_bound = mean-2*std
outlier_count = len(noextreme['y'][noextreme['y'] > upper_bound])

#plot
plt.axvline(x=upper_bound, color='r', linestyle='-')
plt.axvline(x=lower_bound, color='b', linestyle='-')
plt.hist(noextreme['y'], bins = 50 )

print('we have {} outliers'.format(outlier_count))
print('we have {} extreme values'.format(countextreme))
print("These are {} values we will investigate      When finding corresponding ids in excel sheet, use id+2".format(outlier_count+countextreme))
df['churn'].nlargest(33)


# In[13]:


# churn(t) vs churn(t-1) plot
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)

df['churn_shifted'] = df.groupby(['clientID_1'])['churn'].shift(1)
plt.scatter(df['churn'],df['churn_shifted'])


# In[14]:


# churn(t) vs churn(t-1) plot by client_ID
 
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neighbors import LocalOutlierFactor
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)

df['churn_shifted'] = df.groupby(['clientID_1'])['churn'].shift(1)

sns.pairplot(x_vars=["churn"], y_vars=["churn_shifted"], data=df, hue="clientID_1", size=5)


# In[12]:


# Anomaly detection with Local Outlier Factor 
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neighbors import LocalOutlierFactor
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)

#make churn(t-1)
df['churn_shifted'] = df.groupby(['clientID_1'])['churn'].shift(1)

#select the non-NaN terms
lofdata = df.loc[~df['churn_shifted'].isna()]
lofdata = pd.DataFrame(lofdata)

#LOF Modeling, -1 is outlier 
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(lofdata[['churn','churn_shifted']])
lofdata['outlier'] = y_pred



#plot
sns.pairplot(x_vars=["churn"], y_vars=["churn_shifted"], data=lofdata, hue="outlier", size=5)
plt.ylabel('churn_previous')
plt.title('Outlier Detection with Local Outlier Factor Model') 


# In[1]:


## locating the outliers with standard deviation method
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)

## delete the big extremes and zeros
noextreme= df['churn'][np.abs(df['churn']-df['churn'].mean())<=(3*df['churn'].std())]
countextreme = len(df['churn'][np.abs(df['churn']-df['churn'].mean())>(3*df['churn'].std())])
noextreme= noextreme[noextreme>0]

#normalize the data using boxcox
noextreme['y'], lam = boxcox(noextreme)

# mean / std for plotting purpose
mean = noextreme['y'].mean()
std =  noextreme['y'].std()
upper_bound = mean+2*std
lower_bound = mean-2*std
outlier_count = len(noextreme['y'][noextreme['y'] > upper_bound])
total_outlier = outlier_count+countextreme

#plot
plt.axvline(x=upper_bound, color='r', linestyle='-')
plt.axvline(x=lower_bound, color='b', linestyle='-')
plt.hist(noextreme['y'], bins = 50 )
print('we have {} outliers'.format(outlier_count))
print('we have {} extreme values'.format(countextreme))
print("These are {} values we will investigate      When finding corresponding ids in excel sheet, use id+2".format(total_outlier))

#give identifiers for outliers
outlierdata = pd.DataFrame(df['churn'].nlargest(33))
outlierdata['outlier_indicator'] = 1

#join them on the dataset
complete_data = pd.DataFrame(df.join(outlierdata.set_index('churn'), on='churn'))
complete_data = complete_data.fillna(0)
complete_data[complete_data['outlier_indicator']==1]

#complete_data.to_csv('complete_data.csv', sep='\t')


# In[17]:


# locate the outliers with Local Outlier Factor 
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neighbors import LocalOutlierFactor
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)

#make churn(t-1)
df['churn_shifted'] = df.groupby(['clientID_1'])['churn'].shift(1)

#select the non-NaN terms
lofdata = df.loc[~df['churn_shifted'].isna()]
lofdata = pd.DataFrame(lofdata)

#LOF Modeling, -1 is outlier 
#for choosing n neighbors check : http://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(lofdata[['churn','churn_shifted']])
lofdata['outlier'] = y_pred

print('with LOF model, we get {} outliers'.format(len(lofdata[lofdata['outlier']<0])))
#lofdata[lofdata['outlier']<1]


#plot
#sns.pairplot(x_vars=["churn"], y_vars=["churn_shifted"], data=lofdata, hue="outlier", size=5)


# In[2]:


# locate the outliers with Local Outlier Factor 
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neighbors import LocalOutlierFactor
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)

#make churn(t-1)
df['churn_shifted'] = df.groupby(['clientID_1'])['churn'].shift(1)

#select the non-NaN terms
lofdata = df.loc[~df['churn_shifted'].isna()]
lofdata = pd.DataFrame(lofdata)

#LOF Modeling, -1 is outlier 1 is inliner
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(lofdata[['churn','churn_shifted']])
lofdata['outlier_indicator_LOF'] = y_pred

print('with LOF model, we get {} outliers'.format(len(lofdata[lofdata['outlier_indicator_LOF']<0])))
#lofdata[lofdata['outlier']<1]

#######################################################################################

## delete the big extremes and zeros
extreme_coefficient = 3
countextreme = len(df['churn'][np.abs(df['churn']-df['churn'].mean())>(extreme_coefficient*df['churn'].std())])
noextreme= df['churn'][np.abs(df['churn']-df['churn'].mean())<=(extreme_coefficient*df['churn'].std())]
noextreme= noextreme[noextreme>0]

#normalize the data using boxcox
noextreme['y'], lam = boxcox(noextreme)

# mean,std,upper and lower bound for plotting purpose
mean = noextreme['y'].mean()
std =  noextreme['y'].std()
outlier_coefficient = 2 
upper_bound = mean+outlier_coefficient*std
lower_bound = mean-outlier_coefficient*std
outlier_count = len(noextreme['y'][noextreme['y'] > upper_bound])
total_outlier = outlier_count+countextreme

#plot
plt.axvline(x=upper_bound, color='r', linestyle='-')
plt.axvline(x=lower_bound, color='b', linestyle='-')
plt.hist(noextreme['y'], bins = 50 )
print('we have {} outliers'.format(outlier_count))
print('we have {} extreme values'.format(countextreme))
print("These are {} values we will investigate      When finding corresponding ids in excel sheet, use id+2".format(total_outlier))

#give identifiers for outliers : -1
outlierdata = pd.DataFrame(df['churn'].nlargest(33))
outlierdata['outlier_indicator_std'] = -1



#join them on the dataset
complete_data = pd.DataFrame(df.join(outlierdata.set_index('churn'), on='churn'))


#if not outlier: 0 
complete_data['outlier_indicator_std'] = complete_data['outlier_indicator_std'].fillna(0)

#concatenate two dataframe
complete_data = pd.concat([complete_data,lofdata], axis=1)

#retrieve only the data we want
outlierdata = pd.DataFrame(complete_data[['clientID_1','date','churn','churn_shifted','outlier_indicator_std','outlier_indicator_LOF']])
disagreements = outlierdata[(outlierdata['outlier_indicator_std']<0)&(outlierdata['outlier_indicator_LOF']>0)]
outlier35 =  outlierdata[outlierdata['outlier_indicator_std']<0]

outlier35
#complete_data
#complete_data['outlier_indicator_LOF'] = 

#complete_data = complete_data.join(lofdata.set_index('churn'), on = 'churn')
#complete_data
#complete_data[complete_data['outlier_indicator_std']==-1]
#complete_data.to_csv('complete_data.csv', sep='\t')



#plot
#sns.pairplot(x_vars=["churn"], y_vars=["churn_shifted"], data=lofdata, hue="outlier", size=5)


# In[19]:


# locate the outliers with Local Outlier Factor 
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neighbors import LocalOutlierFactor
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)

#make churn(t-1)
df['churn_shifted'] = df.groupby(['clientID_1'])['churn'].shift(1)

#select the non-NaN terms
lofdata = df.loc[~df['churn_shifted'].isna()]
lofdata = pd.DataFrame(lofdata)

#LOF Modeling, -1 is outlier 1 is inliner
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(lofdata[['churn','churn_shifted']])
lofdata['outlier_indicator_LOF'] = y_pred

print('with LOF model, we get {} outliers'.format(len(lofdata[lofdata['outlier_indicator_LOF']<0])))
#lofdata[lofdata['outlier']<1]

#######################################################################################

## delete the big extremes and zeros
extreme_coefficient = 3
countextreme = len(df['churn'][np.abs(df['churn']-df['churn'].mean())>(extreme_coefficient*df['churn'].std())])
noextreme= df['churn'][np.abs(df['churn']-df['churn'].mean())<=(extreme_coefficient*df['churn'].std())]
noextreme= noextreme[noextreme>0]

#normalize the data using boxcox
noextreme['y'], lam = boxcox(noextreme)

# mean,std,upper and lower bound for plotting purpose
mean = noextreme['y'].mean()
std =  noextreme['y'].std()
outlier_coefficient = 2 
upper_bound = mean+outlier_coefficient*std
lower_bound = mean-outlier_coefficient*std
outlier_count = len(noextreme['y'][noextreme['y'] > upper_bound])
total_outlier = outlier_count+countextreme

#plot
#plt.axvline(x=upper_bound, color='r', linestyle='-')
#plt.axvline(x=lower_bound, color='b', linestyle='-')
#plt.hist(noextreme['y'], bins = 50 )
#print('we have {} outliers'.format(outlier_count))
#print('we have {} extreme values'.format(countextreme))
print("These are {} values we will investigate      When finding corresponding ids in excel sheet, use id+2".format(total_outlier))

#give identifiers for outliers : -1
outlierdata = pd.DataFrame(df['churn'].nlargest(33))
outlierdata['outlier_indicator_std'] = -1

#join them on the dataset
complete_data = pd.DataFrame(df.join(outlierdata.set_index('churn'), on='churn'))

#if not outlier: 0 
complete_data['outlier_indicator_std'] = complete_data['outlier_indicator_std'].fillna(0)

#concatenate two dataframe
complete_data = pd.concat([complete_data,lofdata], axis=1)

#retrieve only the data we want
outlierdata = pd.DataFrame(complete_data[['clientID_1','date','churn','churn_shifted','outlier_indicator_std','outlier_indicator_LOF']])
disagreements = outlierdata[(outlierdata['outlier_indicator_std']<0)&(outlierdata['outlier_indicator_LOF']>0)]
outlier35 =  outlierdata[outlierdata['outlier_indicator_std']<0]

outlier35


# In[1]:


# locate the outliers with Local Outlier Factor 
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neighbors import LocalOutlierFactor
from dateutil.relativedelta import relativedelta
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)

#make churn(t-1)
df['churn_shifted'] = df.groupby(['clientID_1'])['churn'].shift(1)

#select the non-NaN terms
lofdata = df.loc[~df['churn_shifted'].isna()]
lofdata = pd.DataFrame(lofdata)

#LOF Modeling, -1 is outlier 1 is inliner
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(lofdata[['churn','churn_shifted']])
lofdata['outlier_indicator_LOF'] = y_pred

#print('with LOF model, we get {} outliers'.format(len(lofdata[lofdata['outlier_indicator_LOF']<0])))
#lofdata[lofdata['outlier']<1]

#######################################################################################

## delete the big extremes and zeros
extreme_coefficient = 3
countextreme = len(df['churn'][np.abs(df['churn']-df['churn'].mean())>(extreme_coefficient*df['churn'].std())])
noextreme= df['churn'][np.abs(df['churn']-df['churn'].mean())<=(extreme_coefficient*df['churn'].std())]
noextreme= noextreme[noextreme>0]

#normalize the data using boxcox
noextreme['y'], lam = boxcox(noextreme)


# mean,std,upper and lower bound for plotting purpose
mean = noextreme['y'].mean()
std =  noextreme['y'].std()
outlier_coefficient = 2 
upper_bound = mean+outlier_coefficient*std
lower_bound = mean-outlier_coefficient*std
outlier_count = len(noextreme['y'][noextreme['y'] > upper_bound])
total_outlier = outlier_count+countextreme

#plot
#plt.axvline(x=upper_bound, color='r', linestyle='-')
#plt.axvline(x=lower_bound, color='b', linestyle='-')
#plt.hist(noextreme['y'], bins = 50 )
#print('we have {} outliers'.format(outlier_count))
#print('we have {} extreme values'.format(countextreme))
#print("These are {} values we will investigate\
      #When finding corresponding ids in excel sheet, use id+2".format(total_outlier))

#give identifiers for outliers : -1
outlierdata = pd.DataFrame(df['churn'].nlargest(33))
outlierdata['outlier_indicator_std'] = -1

#join them on the dataset
complete_data = pd.DataFrame(df.join(outlierdata.set_index('churn'), on='churn'))

#if not outlier: 0 
complete_data['outlier_indicator_std'] = complete_data['outlier_indicator_std'].fillna(0)

#concatenate two dataframe (&& ALERT)
complete_data = pd.concat([complete_data,lofdata], axis=1)

#retrieve only the data we want
outlierdata = pd.DataFrame(complete_data[['clientID_1','date','numberChurned','numberExisting','churn','churn_shifted','outlier_indicator_std','outlier_indicator_LOF']])
disagreements = outlierdata[(outlierdata['outlier_indicator_std']<0)&(outlierdata['outlier_indicator_LOF']>0)]
outliers = pd.DataFrame(outlierdata[(outlierdata['outlier_indicator_std']<0)&(outlierdata['outlier_indicator_LOF']<0)])
outliers = outliers.loc[:,~outliers.columns.duplicated()]
outliers = outliers.drop_duplicates(keep='first') 

conditions = [
    (outliers['numberExisting'] <= 100),
    (outliers['numberExisting'] <= 5000)]
choices = ['Micro', 'Large']
outliers['size'] = np.select(conditions, choices, default='Mega')
outliers['date'] = pd.to_datetime(outliers['date'])
outliers['enddate'] = outliers['date'] + timedelta(days=30)

outliers.to_csv('outliers.csv', sep='\t')
outliers.to_csv('outliers.csv', sep='\t', encoding='utf-8')
outliers


# In[2]:


# locate systematic churn percentages
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neighbors import LocalOutlierFactor
df = pd.read_csv('C:/Users\dhuh\Downloads\wac1_churn.csv', names = ["clientID", "planID", "cusID", "updDate","accessID","Null"])

df['updDate'] = pd.to_datetime(df['updDate'])
#df['updDate'] = pd.Timestamp(df['updDate'])


#make updDate(t-1)
df['updDate_shifted'] = df.groupby(['planID'])['updDate'].shift(1)
df['time_difference'] = (df['updDate']-df['updDate_shifted'])
df['systematic_churn'] = np.where(df['time_difference']<='0 days 00:00:10.000000', 1, 0)
churn_sum = df['systematic_churn'].sum()
count = len(df['systematic_churn'])
proportion= churn_sum/count
proportion


# In[3]:


import pymssql
or_list = []

for index, row in outliers.iterrows():
   or_list.append('(acim.clientID = \'%s\' AND feh.updDate Between \'%s\' and \'%s\' )' 
                  % (row['clientID_1'].strip(), row['date'], row['enddate']))


or_condition = ' OR '.join(or_list)

#print(or_condition)


sql_query = 'SELECT acim.clientID, acim.planID, acim.cusID, feh.updDate,feh.accessID FROM dbo.fullEnrollmentHistory AS feh WITH (NOLOCK) INNER JOIN dbo.accountCusIDMap AS acim WITH (NOLOCK) ON acim.accountID = feh.accountID WHERE ' + or_condition + ' and feh.enroll = 0 order by planID, feh.updDate'

conn = pymssql.connect(server='work-wfeio-daily.cp1ziqpczqez.us-east-1.rds.amazonaws.com', user='Analytics', password='@n@lytic5')

df_practice = pd.read_sql(sql_query, conn)
df_practice

conn.close()
# #work-wfeio-daily.cp1ziqpczqez.us-east-1.rds.amazonaws.com


# In[245]:


from pandas.tseries.offsets import MonthBegin

df_practice['rounded_time'] = df_practice['updDate'].dt.round('D') 

df_practice['rounded_time'] = df_practice['rounded_time'] - MonthBegin(n=1)
df_practice
df_practice.groupby(['rounded_time','clientID']).agg({'cusID': pd.Series.nunique})


# In[167]:


# locate the outliers with Local Outlier Factor 
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neighbors import LocalOutlierFactor
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)

#make churn(t-1)
df['churn_shifted'] = df.groupby(['clientID_1'])['churn'].shift(1)

#select the non-NaN terms
lofdata = df.loc[~df['churn_shifted'].isna()]
lofdata = pd.DataFrame(lofdata)

#LOF Modeling, -1 is outlier 1 is inliner
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(lofdata[['churn','churn_shifted']])
lofdata['outlier_indicator_LOF'] = y_pred

#print('with LOF model, we get {} outliers'.format(len(lofdata[lofdata['outlier_indicator_LOF']<0])))
#lofdata[lofdata['outlier']<1]

#######################################################################################

## delete the big extremes and zeros
extreme_coefficient = 3
countextreme = len(df['churn'][np.abs(df['churn']-df['churn'].mean())>(extreme_coefficient*df['churn'].std())])
noextreme= df['churn'][np.abs(df['churn']-df['churn'].mean())<=(extreme_coefficient*df['churn'].std())]
noextreme= noextreme[noextreme>0]

#normalize the data using boxcox
noextreme['y'], lam = boxcox(noextreme)

# mean,std,upper and lower bound for plotting purpose
mean = noextreme['y'].mean()
std =  noextreme['y'].std()
outlier_coefficient = 2 
upper_bound = mean+outlier_coefficient*std
lower_bound = mean-outlier_coefficient*std
outlier_count = len(noextreme['y'][noextreme['y'] > upper_bound])
total_outlier = outlier_count+countextreme

#plot
#plt.axvline(x=upper_bound, color='r', linestyle='-')
#plt.axvline(x=lower_bound, color='b', linestyle='-')
#plt.hist(noextreme['y'], bins = 50 )
#print('we have {} outliers'.format(outlier_count))
#print('we have {} extreme values'.format(countextreme))
#print("These are {} values we will investigate\
      #When finding corresponding ids in excel sheet, use id+2".format(total_outlier))

#give identifiers for outliers : -1
outlierdata = pd.DataFrame(df['churn'].nlargest(33))
outlierdata['outlier_indicator_std'] = -1

#join them on the dataset
complete_data = pd.DataFrame(df.join(outlierdata.set_index('churn'), on='churn'))

#if not outlier: 0 
complete_data['outlier_indicator_std'] = complete_data['outlier_indicator_std'].fillna(0)

#concatenate two dataframe (&& ALERT)
complete_data = pd.concat([complete_data,lofdata], axis=1)

#retrieve only the data we want
outlierdata = pd.DataFrame(complete_data[['clientID_1','date','numberChurned','numberExisting','churn','churn_shifted','outlier_indicator_std','outlier_indicator_LOF']])
disagreements = outlierdata[(outlierdata['outlier_indicator_std']<0)&(outlierdata['outlier_indicator_LOF']>0)]
outliers = pd.DataFrame(outlierdata[(outlierdata['outlier_indicator_std']<0)&(outlierdata['outlier_indicator_LOF']<0)])
outliers = outliers.loc[:,~outliers.columns.duplicated()]
outliers = outliers.drop_duplicates(keep='first') 

conditions = [
    (outliers['numberExisting'] <= 100),
    (outliers['numberExisting'] <= 5000)]
choices = ['Micro', 'Large']
outliers['size'] = np.select(conditions, choices, default='Mega')
outliers['date'] = pd.to_datetime(outliers['date'])
outliers['enddate'] = outliers['date'] + timedelta(days=30)

outliers.to_csv('outliers.csv', sep='\t')
outliers.to_csv('outliers.csv', sep='\t', encoding='utf-8')
outliers['numberChurned'].sum()
outliers
#outliers


# In[4]:


import pymssql
or_list = []

for index, row in outliers.iterrows():
   or_list.append('(acim.clientID = \'%s\' AND feh.updDate Between \'%s\' and \'%s\' )' 
                  % (row['clientID_1'].strip(), row['date'], row['enddate']))


or_condition = ' OR '.join(or_list)

#print(or_condition)


sql_query = 'SELECT acim.clientID, acim.planID, acim.cusID, feh.updDate,feh.accessID FROM dbo.fullEnrollmentHistory AS feh WITH (NOLOCK) INNER JOIN dbo.accountCusIDMap AS acim WITH (NOLOCK) ON acim.accountID = feh.accountID WHERE ' + or_condition + ' and feh.enroll = 0 order by planID, feh.updDate'



conn = pymssql.connect(server='work-wfeio-daily.cp1ziqpczqez.us-east-1.rds.amazonaws.com', user='Analytics', password='@n@lytic5')

print('Querying data..........')

df_whole = pd.read_sql(sql_query, conn)
df_whole

conn.close()
# #work-wfeio-daily.cp1ziqpczqez.us-east-1.rds.amazonaws.com


# In[6]:


#calculate MaxPercentage

df_whole.dtypes

# get new column that truncates updDate to datetime without timestamp
df_whole['updDateDay'] = df_whole['updDate'].dt.date
df_whole['updDateMonth'] = df_whole['updDate'].dt.to_period('M')

groupforsum= ['clientID', 'updDateMonth','updDateDay']
toGroupBy  = ['clientID', 'updDateDay']
# write function that uses groupby and a list of columns you want to groupby
# groupby count distinct cusID
tmp = df_whole.groupby(toGroupBy, as_index=False).agg({'cusID':pd.Series.nunique})
#grp = df_whole.groupby(groupforsum, as_index=False).agg({'cusID':pd.Series.nunique})


#tmp
# take aggregate dataframe add new column that is month-start of updDate (one of the cols you just grouped by)
tmp['updDateDay'] = pd.to_datetime(tmp['updDateDay'] )
tmp['updDateMonth'] = tmp['updDateDay'].dt.to_period('M')

#tmp.groupby

#tmp
#g= tmp.groupby(['clientID','updDateMonth']).sum()
#tmp
#k = df_whole.groupby(['clientID','updDateMonth'], as_index=False)['cusID'].transform(lambda series: series / series.sum())
tmp['MaxPercentage'] = tmp.groupby(['clientID','updDateMonth'], as_index=False)['cusID'].transform(lambda x: x*100/ x.sum())
tmp.sort_values(['updDateDay','clientID'])
idx = tmp.groupby(['clientID','updDateMonth'])['MaxPercentage'].transform(max) == tmp['MaxPercentage']
highpercentage = pd.DataFrame(tmp[idx])
highpercentage = highpercentage.sort_values(['updDateMonth','clientID'])
highpercentage = highpercentage[['clientID','updDateMonth','updDateDay','MaxPercentage','cusID']]
highpercentage = highpercentage.rename(columns={'cusID': 'CountCusID', 'updDateDay' : 'MaxChurnDate'})

highpercentage


# In[8]:


####GOLD


highpercentage.reset_index(drop=True, inplace=True)
outliers.reset_index(drop=True, inplace=True)

highpercentage = pd.concat([highpercentage,outliers[['churn','numberChurned']]],axis=1)

highpercentage = highpercentage[['clientID','updDateMonth','churn','numberChurned','MaxChurnDate','MaxPercentage']]
highpercentage = highpercentage.loc[:,~highpercentage.columns.duplicated()]

highpercentage.sort_values(['MaxPercentage','churn'], ascending=False)


# In[157]:


df_whole.head()


# In[247]:


# Churn window for SEBE-DST
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neighbors import LocalOutlierFactor
df = pd.read_csv('C:/Users\dhuh\Downloads\trpx_churn.csv', names = ["cusID","clientID", "planID", "updDate","cusID"])

df['updDate'] = pd.to_datetime(df['updDate'])
#df['updDate'] = pd.Timestamp(df['updDate'])


#make updDate(t-1)
df['updDate_shifted'] = df.groupby(['planID'])['updDate'].shift(1)
df['time_difference'] = (df['updDate']-df['updDate_shifted'])
df['systematic_churn'] = np.where(df['time_difference']<='0 days 00:00:10.000000', 1, 0)

df['rounded_time'] = df['updDate'].dt.round('D') 

df.groupby(['clientID','rounded_time']).agg({'cusID': pd.Series.nunique})
bydate =pd.DataFrame(df.groupby(['clientID','rounded_time']).agg({'cusID': pd.Series.nunique}))
bydate['percentage(%)'] = (bydate['cusID'])*100/(bydate['cusID'].sum())

#print(bydate.sum())
bydate


#df.groupby('planID').agg({'rounded_time': pd.Series.nunique})



#churn_sum = df['systematic_churn'].sum()
#count = len(df['systematic_churn'])
#proportion= churn_sum/count
#proportion


# In[106]:


# locate the outliers with Local Outlier Factor 
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neighbors import LocalOutlierFactor
from dateutil.relativedelta import relativedelta
df = pd.read_csv('C:/Users\dhuh\Downloads\churnStats2013to2018-06-01clientID_1MS.csv', header = 0)

#make churn(t-1)
df['churn_shifted'] = df.groupby(['clientID_1'])['churn'].shift(1)

#select the non-NaN terms
lofdata = df.loc[~df['churn_shifted'].isna()]
lofdata = pd.DataFrame(lofdata)

#LOF Modeling, -1 is outlier 1 is inliner
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(lofdata[['churn','churn_shifted']])
lofdata['outlier_indicator_LOF'] = y_pred

#print('with LOF model, we get {} outliers'.format(len(lofdata[lofdata['outlier_indicator_LOF']<0])))
#lofdata[lofdata['outlier']<1]

#######################################################################################

## delete the big extremes and zeros
extreme_coefficient = 3
countextreme = len(df['churn'][np.abs(df['churn']-df['churn'].mean())>(extreme_coefficient*df['churn'].std())])
noextreme= df['churn'][np.abs(df['churn']-df['churn'].mean())<=(extreme_coefficient*df['churn'].std())]
noextreme= noextreme[noextreme>0]

#normalize the data using boxcox
noextreme['y'], lam = boxcox(noextreme)


# mean,std,upper and lower bound for plotting purpose
mean = noextreme['y'].mean()
std =  noextreme['y'].std()
outlier_coefficient = 2 
upper_bound = mean+outlier_coefficient*std
lower_bound = mean-outlier_coefficient*std
outlier_count = len(noextreme['y'][noextreme['y'] > upper_bound])
total_outlier = outlier_count+countextreme

#plot
#plt.axvline(x=upper_bound, color='r', linestyle='-')
#plt.axvline(x=lower_bound, color='b', linestyle='-')
#plt.hist(noextreme['y'], bins = 50 )
#print('we have {} outliers'.format(outlier_count))
#print('we have {} extreme values'.format(countextreme))
#print("These are {} values we will investigate\
      #When finding corresponding ids in excel sheet, use id+2".format(total_outlier))

#give identifiers for outliers : -1
outlierdata = pd.DataFrame(df['churn'].nlargest(33))
outlierdata['outlier_indicator_std'] = -1

#join them on the dataset
complete_data = pd.DataFrame(df.join(outlierdata.set_index('churn'), on='churn'))

#if not outlier: 0 
complete_data['outlier_indicator_std'] = complete_data['outlier_indicator_std'].fillna(0)

#concatenate two dataframe (&& ALERT)
complete_data = pd.concat([complete_data,lofdata], axis=1)

#retrieve only the data we want
outlierdata = pd.DataFrame(complete_data[['clientID_1','date','numberChurned','numberExisting','churn','churn_shifted','outlier_indicator_std','outlier_indicator_LOF']])
disagreements = outlierdata[(outlierdata['outlier_indicator_std']<0)&(outlierdata['outlier_indicator_LOF']>0)]
outliers = pd.DataFrame(outlierdata[(outlierdata['outlier_indicator_std']<0)&(outlierdata['outlier_indicator_LOF']<0)])
outliers = outliers.loc[:,~outliers.columns.duplicated()]
outliers = outliers.drop_duplicates(keep='first') 

conditions = [
    (outliers['numberExisting'] <= 100),
    (outliers['numberExisting'] <= 5000)]
choices = ['Micro', 'Large']
outliers['size'] = np.select(conditions, choices, default='Mega')
outliers['date'] = pd.to_datetime(outliers['date'])
outliers['enddate'] = outliers['date'] + timedelta(days=30)

outliers.to_csv('outliers.csv', sep='\t')
outliers.to_csv('outliers.csv', sep='\t', encoding='utf-8')
outliers


# In[242]:


# churn window for WAC1
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neighbors import LocalOutlierFactor
df = pd.read_csv('C:/Users\dhuh\Downloads\trpx_churn.csv', names = ["clientID", "planID", "cusID", "updDate","accessID","Null"])

df['updDate'] = pd.to_datetime(df['updDate'])
#df['updDate'] = pd.Timestamp(df['updDate'])


#make updDate(t-1)
df['updDate_shifted'] = df.groupby(['planID'])['updDate'].shift(1)
df['time_difference'] = (df['updDate']-df['updDate_shifted'])
df['systematic_churn'] = np.where(df['time_difference']<='0 days 00:00:10.000000', 1, 0)

df['rounded_time'] = df['updDate'].dt.round('D') 

bydate =pd.DataFrame(df.groupby(['clientID','rounded_time']).agg({'cusID': pd.Series.nunique}))
bydate['percentage(%)'] = (bydate['cusID'])*100/(bydate['cusID'].sum())

bydate
#bydate['percentage(%)'].max()
#bydate.loc[bydate['percentage(%)'] == bydate['percentage(%)'].max() ]

#proportion


# In[244]:


# Churn window for df_whole
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import math 
from pandas import Series, DataFrame, Panel
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neighbors import LocalOutlierFactor

df_whole

df_whole['rounded_time'] = df_whole['updDate'].dt.round('D') 

df_whole.groupby(['clientID','rounded_time']).agg({'cusID': pd.Series.nunique})
bydate =pd.DataFrame(df_whole.groupby(['clientID','rounded_time']).agg({'cusID': pd.Series.nunique}))
bydate['percentage(%)'] = (bydate['cusID'])*100/(bydate['cusID'].sum())

#print(bydate.sum())
bydate

#bydate.loc[bydate['clientID']=='TRPX']

#df.groupby('planID').agg({'rounded_time': pd.Series.nunique})



#churn_sum = df['systematic_churn'].sum()
#count = len(df['systematic_churn'])
#proportion= churn_sum/count
#proportion


# In[11]:


df_whole.head()

