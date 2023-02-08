#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_excel('assignment_order_masking.xlsx')


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.head(10)


# In[6]:


df.describe()


# In[7]:


df.groupby('ord_status')['sl_no'].count()


# In[8]:


# ord_status is skewed towards 1, this can impact our model and in this case we need more data


# In[9]:


df.nunique(axis=0)


# In[10]:


sns.heatmap(df.isnull())


# In[11]:


df.isna().mean().round(4) * 100


# In[12]:


# Lot of missing values in column buffer,cancellation_perc,cf_ratio,buffer_to_soh so we have to drop these columns from our model


# In[13]:


features_with_na = [features for features in df.columns if df[features].isnull().sum() > 1]


# In[14]:


for feature in features_with_na:
    data = df.copy()
    data[feature] = np.where(data[feature].isnull(),1,0)
    
    data.groupby(feature)['ord_status'].mean().plot.bar()
    plt.title(feature)
    plt.show()


# In[15]:


# Relationship with missing variable and dependent variable is clearly visible so we should replace these missing values with something meaningful but missing percentages are high


# In[16]:


numerical_features = [feature for feature in df.columns if df[feature].dtypes !='O']
len(numerical_features)


# In[17]:


discrete_features = [feature for feature in df.columns if (len(df[feature].unique()) < 25 and feature != 'sl_no') or (df[feature].dtype != 'float64' and feature != 'sl_no')  ]
discrete_features


# In[18]:


for feature in discrete_features:
    data = df.copy()
    #data.groupby(feature)['ord_status'].mean().plot.bar()
    try:
        sns.distplot(data[feature])
        plt.axhline(0.5, color='black')
        plt.xlabel(feature)
        plt.ylabel('ord_status')
        plt.title(feature)
        plt.show()
    except:
        pass


# In[19]:


# skewed order_lag_in_hours and soh


# In[20]:


for feature in discrete_features:
    data = df.copy()
    
    try:
        data.groupby(feature)['ord_status'].mean().plot.bar()
        plt.axhline(0.5, color='black')
        plt.xlabel(feature)
        plt.ylabel('ord_status')
        plt.title(feature)
        plt.show()
    except:
        pass


# In[21]:


# We can see that season - winter has lowest impact on ord_status (product to mask)
# State sth has more impact on ord_status - (Product to sell)
# Rest variables are more or less inconclusive


# In[22]:



continous_feature = [feature for feature in df.columns if feature not in discrete_features and feature != 'sl_no']
continous_feature


# In[23]:



for feature in continous_feature:
    data = df.copy()
    try:
        sns.distplot(data[feature])
        plt.xlabel(feature)
        plt.ylabel('ord_status')
        plt.title(feature)
        plt.show()
    except:
        pass


# In[24]:


# columns - envelope_avg_days_to_sale and volumne columns are very skewed, shows outliers so we will handle it in feature engineering


# In[25]:


# It's a skewed data - so we have to do normalization


# In[26]:


# Outlier in numerical Features


# In[27]:


for feature in numerical_features:
    print(feature)
    try :
        sns.boxplot(df[feature])
        plt.show()
    except:
        pass


# In[28]:


#Outliers In Numerical Features - soh,buffer,order_lag_in_hours,envelope_avg_days_to_sale,cf_ratio,volume_in_cm3,buffer_to_soh


# In[29]:



categorical_feature = [feature for feature in df.columns if df[feature].dtypes == 'O']
categorical_feature


# In[30]:


for feature in categorical_feature:
    data = df.copy()
    data.groupby(feature)['ord_status'].mean().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('ord_status')
    plt.title(feature)
    plt.show()


# In[ ]:




