#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#missing value columns are buffer,cancellation_perc,cf_ratio,volume_in_cm3,buffer_to_soh                


# In[3]:


# All of missing value columns are numeric - So we can replace the missing values with median for columns with high outliers among these


# In[4]:


#missing_columns = ['buffer','cancellation_perc','cf_ratio','volume_in_cm3','buffer_to_soh']


# In[5]:


"""for feature in missing_columns:
    median_value = df[feature].median()
    for i in range(len(df)):
        if(df[feature].iloc[i].dtype == 'float64' or df[feature].iloc[i].dtype == 'int64'):
            pass
        else:
            df[feature].iloc[i] = median_value
            """


# In[6]:


df.drop(['buffer','cancellation_perc','cf_ratio','volume_in_cm3','buffer_to_soh'],axis=1,inplace=True)
df.head()


# In[7]:


df.drop(['sl_no','online_exclusive'],axis=1,inplace=True)


# In[8]:


outliers = ['soh','order_lag_in_hours','envelope_avg_days_to_sale']


# In[9]:


for feature in outliers:
    q1 = df[feature].quantile(0.25)
    q2 = df[feature].quantile(0.75)
    IQR = q2 - q1
    max_limit = q2 + (1.5 * IQR)
    min_limit = q1 - (1.5 * IQR) 
    df[feature] = np.where(df[feature] > max_limit, max_limit, 
             (np.where(df[feature] < min_limit, min_limit, df[feature])))


# In[10]:


for feature in outliers:
    try :
        sns.boxplot(df[feature])
        plt.show()
    except:
        pass


# In[11]:


"""Why is skewness a problem?
Most of the statistical models do not work when the data is skewed. The reason behind this is that the tapering ends or the tail region of the skewed data distributions are the outliers in the data and it is known to us that outliers can severely damage the performance of a statistical model. The best example of this being regression models that show bad results when trained over skewed data."""


# In[12]:


skewed_features = ['soh','order_lag_in_hours','envelope_avg_days_to_sale']


# In[13]:


""" Algorithms tend to perform better or converge faster when the different features (variables) are on a smaller scale. Therefore it is common practice to normalize the data before training machine learning models on it.
Normalization also makes the training process less sensitive to the scale of the features. This results in getting better coefficients after training."""


# In[14]:


from sklearn.preprocessing import PowerTransformer, QuantileTransformer


# In[15]:



pt = PowerTransformer()
qt = QuantileTransformer(n_quantiles=500, output_distribution='normal')


# In[16]:


for feature in skewed_features:
    mean = df[feature].mean()
    std = df[feature].std()
    median = df[feature].median()
    array = np.array(df[feature]).reshape(-1, 1)
    df[feature] = qt.fit_transform(array)
    print(df[feature].describe())
    sns.distplot(df[feature])
    plt.show()


# In[17]:


state_dummies = pd.get_dummies(df['state'])
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(state_dummies.corr(),annot=True)
plt.show()


# In[18]:


seasonality_dummies = pd.get_dummies(df['seasonality'],drop_first=True)
sns.heatmap(seasonality_dummies.corr(),annot=True)


# In[19]:


"""Three benefits of performing feature selection before modeling your data are:
Reduces Overfitting: Less redundant data means less opportunity to make decisions based on noise.
Improves Accuracy: Less misleading data means modeling accuracy improves.
Reduces Training Time: Less data means that algorithms train faster."""


# In[20]:


# Removing state and seasonality column with their dummy columns


# In[21]:


df = pd.concat([df,state_dummies,seasonality_dummies],axis=1)


# In[22]:


df.drop(['state','seasonality'],axis =1,inplace = True)


# In[23]:


# Feature Scaling


# In[24]:


from sklearn.preprocessing import StandardScaler


# In[26]:


scaler = StandardScaler()
scaler.fit(df.drop('ord_status',axis=1))
scaled_features = scaler.transform(df.drop('ord_status',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[1:])


# In[ ]:




