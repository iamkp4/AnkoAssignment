#!/usr/bin/env python
# coding: utf-8

# In[1]:


fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(df_feat.corr(),annot=True)
plt.show()


# In[2]:


# Dropping columns with high corr - envelope_avg_days_to_sale,store_avg_pick_tat


# In[3]:


df_feat.drop(['envelope_avg_days_to_sale','store_avg_pick_tat'],axis=1,inplace=True)


# In[4]:



X = df_feat
Y = df['ord_status']
X.columns


# In[5]:



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3,random_state=0)


# In[6]:


from sklearn.feature_selection import SelectFromModel


# In[7]:


import sklearn


# In[8]:


feature_sel_model = SelectFromModel(sklearn.linear_model.Lasso(alpha=0.001,random_state=0))
feature_sel_model.fit(X_train,y_train)
feature_sel_model.get_support()


# In[9]:


# So dropping nth column and order_lag_in_hours


# In[10]:


# Resampling the Data


# In[11]:


from imblearn.under_sampling import RandomUnderSampler


# In[12]:


X = df_feat.drop(['nth','order_lag_in_hours'],axis=1)
Y = df['ord_status']
X.columns


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size = 0.3,random_state=0)


# In[13]:


rus = RandomUnderSampler()
X_res,y_res = rus.fit_resample(X_train,y_train)


# In[14]:


from collections import Counter
Counter(y_res)


# In[15]:


X_train = X_res
y_train = y_res


# In[ ]:




