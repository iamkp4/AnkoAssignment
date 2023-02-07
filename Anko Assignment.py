#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_excel('assignment_order_masking.xlsx')


# # Exploratory Data Analysis

# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.head(10)


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.groupby('ord_status')['sl_no'].count()


# In[ ]:


# ord_status is skewed towards 1, this can impact our model and in this case we need more data


# In[9]:


df.nunique(axis=0)


# In[11]:


#online_exculsive has 1 unique value so it won't impact the model so we will drop this column


# In[10]:


sns.heatmap(df.isnull())


# In[12]:


df.isna().mean().round(4) * 100


# In[13]:


# Lot of missing values in column buffer,cancellation_perc,cf_ratio,buffer_to_soh so we have to drop these columns from our model


# In[14]:


features_with_na = [features for features in df.columns if df[features].isnull().sum() > 1]


# In[15]:


features_with_na


# In[16]:


# Seeing if missing values ( rows) has any impact on the ord_status


# In[17]:


for feature in features_with_na:
    data = df.copy()
    data[feature] = np.where(data[feature].isnull(),1,0)
    
    data.groupby(feature)['ord_status'].mean().plot.bar()
    plt.title(feature)
    plt.show()


# In[19]:


# Relationship with missing variable and dependent variable is clearly visible so we should replace these missing values with something meaningful but missing percentages are high


# In[18]:


numerical_features = [feature for feature in df.columns if df[feature].dtypes !='O']
len(numerical_features)


# In[19]:


discrete_features = [feature for feature in df.columns if (len(df[feature].unique()) < 25 and feature != 'sl_no') or (df[feature].dtype != 'float64' and feature != 'sl_no')  ]
discrete_features


# In[20]:


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


# In[ ]:


# skewed order_lag_in_hours and soh


# In[25]:


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


# In[ ]:


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


# In[28]:


# columns - envelope_avg_days_to_sale and volumne columns are very skewed, shows outliers so we will handle it in feature engineering


# In[29]:


# It's a skewed data - so we have to do normalization


# In[30]:


from sklearn import preprocessing


# In[31]:


for feature in continous_feature:
    data = df.copy()
    try:
        normalized = preprocessing.normalize([data[feature]])
        sns.distplot(normalized)
        plt.xlabel(feature)
        plt.ylabel('ord_status')
        plt.title(feature)
        plt.show()
    except:
        pass


# # Finding Outliers

# In[24]:


df.isna().mean().round(4) * 100


# In[25]:


# Outlier in numerical Features


# In[26]:


numerical_features


# In[27]:


for feature in numerical_features:
    print(feature)
    try :
        sns.boxplot(df[feature])
        plt.show()
    except:
        pass


# In[ ]:


#Outliers In Numerical Features - soh,buffer,order_lag_in_hours,envelope_avg_days_to_sale,cf_ratio,volume_in_cm3,buffer_to_soh


# # Categorical Features

# In[28]:


categorical_feature = [feature for feature in df.columns if df[feature].dtypes == 'O']
categorical_feature


# In[29]:


for feature in categorical_feature:
    print(feature," :",len(df[feature].unique()))


# In[30]:


for feature in categorical_feature:
    data = df.copy()
    data.groupby(feature)['ord_status'].mean().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('ord_status')
    plt.title(feature)
    plt.show()


# # Feature Engineering

# In[31]:


#missing value columns are buffer,cancellation_perc,cf_ratio,volume_in_cm3,buffer_to_soh                


# In[32]:


# All of missing value columns are numeric - So we can replace the missing values with median for columns with high outliers among these


# In[33]:


#missing_columns = ['buffer','cancellation_perc','cf_ratio','volume_in_cm3','buffer_to_soh']


# In[34]:


"""for feature in missing_columns:
    median_value = df[feature].median()
    for i in range(len(df)):
        if(df[feature].iloc[i].dtype == 'float64' or df[feature].iloc[i].dtype == 'int64'):
            pass
        else:
            df[feature].iloc[i] = median_value
            """


# In[35]:


#df.isna().mean().round(4) * 100


# In[36]:


df.drop(['buffer','cancellation_perc','cf_ratio','volume_in_cm3','buffer_to_soh'],axis=1,inplace=True)
df.head()


# In[37]:


# Dropping columns - sl_no,online_exclusive as they are not helpful in our model


# In[38]:


df.drop(['sl_no','online_exclusive'],axis=1,inplace=True)


# In[39]:


df.head()


# In[40]:


#Handling Outliers


# In[41]:


outliers = ['soh','order_lag_in_hours','envelope_avg_days_to_sale']


# In[43]:


for feature in outliers:
    q1 = df[feature].quantile(0.25)
    q2 = df[feature].quantile(0.75)
    IQR = q2 - q1
    max_limit = q2 + (1.5 * IQR)
    min_limit = q1 - (1.5 * IQR) 
    df[feature] = np.where(df[feature] > max_limit, max_limit, 
             (np.where(df[feature] < min_limit, min_limit, df[feature])))


# In[44]:


for feature in outliers:
    try :
        sns.boxplot(df[feature])
        plt.show()
    except:
        pass


# In[45]:


# Now normalizing the skewed numerical features


# In[46]:


"""Why is skewness a problem?
Most of the statistical models do not work when the data is skewed. The reason behind this is that the tapering ends or the tail region of the skewed data distributions are the outliers in the data and it is known to us that outliers can severely damage the performance of a statistical model. The best example of this being regression models that show bad results when trained over skewed data."""


# In[47]:


skewed_features = ['soh','order_lag_in_hours','envelope_avg_days_to_sale']


# In[48]:


""" Algorithms tend to perform better or converge faster when the different features (variables) are on a smaller scale. Therefore it is common practice to normalize the data before training machine learning models on it.
Normalization also makes the training process less sensitive to the scale of the features. This results in getting better coefficients after training."""


# In[49]:


from sklearn.preprocessing import PowerTransformer, QuantileTransformer


# In[50]:


pt = PowerTransformer()
qt = QuantileTransformer(n_quantiles=500, output_distribution='normal')


# In[51]:


for feature in skewed_features:
    mean = df[feature].mean()
    std = df[feature].std()
    median = df[feature].median()
    array = np.array(df[feature]).reshape(-1, 1)
    df[feature] = qt.fit_transform(array)
    print(df[feature].describe())
    sns.distplot(df[feature])
    plt.show()
    

        


# In[52]:


df.head()


# In[53]:


len(df)


# In[54]:


state_dummies = pd.get_dummies(df['state'])
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(state_dummies.corr(),annot=True)
plt.show()


# In[55]:


seasonality_dummies = pd.get_dummies(df['seasonality'],drop_first=True)
sns.heatmap(seasonality_dummies.corr(),annot=True)


# # Feature Selection

# In[56]:


"""Three benefits of performing feature selection before modeling your data are:
Reduces Overfitting: Less redundant data means less opportunity to make decisions based on noise.
Improves Accuracy: Less misleading data means modeling accuracy improves.
Reduces Training Time: Less data means that algorithms train faster."""


# In[57]:


df.head(10)


# In[58]:


# Removing state and seasonality column with their dummy columns


# In[59]:


df = pd.concat([df,state_dummies,seasonality_dummies],axis=1)


# In[60]:


df.head()


# In[61]:


df.drop(['state','seasonality'],axis =1,inplace = True)
df.head()


# In[62]:


# Feature Scaling


# In[63]:


from sklearn.preprocessing import StandardScaler


# In[64]:


scaler = StandardScaler()


# In[65]:


scaler.fit(df.drop('ord_status',axis=1))


# In[66]:


scaled_features = scaler.transform(df.drop('ord_status',axis=1))


# In[67]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[1:])


# In[69]:


df_feat.head()


# In[70]:


fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(df_feat.corr(),annot=True)
plt.show()


# In[71]:


# Dropping columns with high corr - envelope_avg_days_to_sale,store_avg_pick_tat


# In[78]:


df_feat.drop(['envelope_avg_days_to_sale','store_avg_pick_tat'],axis=1,inplace=True)
df_feat.head()


# In[171]:


X = df_feat
Y = df['ord_status']
X.columns


# In[172]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3,random_state=0)


# In[173]:


len(X_train.columns)


# In[78]:


from sklearn.feature_selection import SelectFromModel


# In[79]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# In[80]:


from numpy import set_printoptions


# In[86]:


"""These options determine the way floating point numbers, arrays and other NumPy objects are displayed."""


# In[83]:


import sklearn


# In[88]:


#test = SelectKBest(score_func=f_classif, k=4)
#fit = test.fit(X_train,y_train)


# In[89]:


#set_printoptions(precision=3)
#print(fit.scores_)
#features = fit.transform(X_train)


# In[174]:


feature_sel_model = SelectFromModel(sklearn.linear_model.Lasso(alpha=0.001,random_state=0))


# In[175]:


feature_sel_model.fit(X_train,y_train)


# In[187]:


X_train.columns


# In[179]:


feature_sel_model.get_support()


# In[94]:


# So dropping nth column and order_lag_in_hours


# In[136]:


# Resampling the Data


# In[182]:


from imblearn.under_sampling import RandomUnderSampler


# In[183]:


X = df_feat.drop(['nth','order_lag_in_hours'],axis=1)
Y = df['ord_status']
X.columns


# In[137]:


X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size = 0.3,random_state=0)


# In[138]:


rus = RandomUnderSampler()


# In[139]:


X_res,y_res = rus.fit_resample(X_train,y_train)


# In[140]:


X_res.shape


# In[141]:


y_res.shape


# In[142]:


from collections import Counter
Counter(y_res)


# In[143]:


X_train = X_res
y_train = y_res


# In[144]:


X_train.columns


# # Model - Machine Learning Model

# In[145]:


from sklearn.ensemble import RandomForestClassifier


# In[146]:


rfc = RandomForestClassifier(n_estimators = 1000)


# In[147]:


rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(mean_squared_error(y_test,predictions))


# In[148]:


plt.hist(predictions)


# In[149]:

# Checking/Comparing with results of LogisticRegression Model

from sklearn.linear_model import LogisticRegression


# In[150]:


logmodel = LogisticRegression()


# In[151]:


logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)


# In[152]:


from sklearn.metrics import classification_report


# In[153]:


print(classification_report(y_test,predictions))


# In[154]:


from sklearn.metrics import confusion_matrix


# In[155]:


confusion_matrix(y_test,predictions)


# In[156]:


probs = logmodel.predict_proba(X)[:, 1] 
plt.hist(probs) 
plt.show()


# In[157]:


score = logmodel.score(X_test,y_test)
print(score)


# In[158]:


from sklearn.metrics import mean_squared_error


# In[159]:


mean_squared_error(y_test,predictions)


# In[160]:


logmodel.coef_


# In[161]:


len(logmodel.coef_[0])


# In[162]:


#Finding Feature Importance


# In[163]:


importance = logmodel.coef_[0]

for i,j in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,j))

plt.bar([X for X in range(len(importance))], importance)
plt.show()


# In[164]:


plt.hist(y_test)


# In[165]:


plt.hist(predictions)


# In[188]:


## Best Fit model is RandomForest with better precision and recall


# In[ ]:




