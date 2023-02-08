#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestClassifier


# In[9]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error


# In[2]:


rfc = RandomForestClassifier(n_estimators = 1000)
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(mean_squared_error(y_test,predictions))


# In[3]:


plt.hist(predictions)


# In[4]:


# Checking for Logistic Regression


# In[5]:


from sklearn.linear_model import LogisticRegression


# In[6]:


logmodel = LogisticRegression()


# In[7]:


logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)


# In[10]:


print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
score = logmodel.score(X_test,y_test)
print(score)
print(mean_squared_error(y_test,predictions))


# In[11]:


#Finding Feature Importance


# In[12]:


importance = logmodel.coef_[0]

for i,j in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,j))

plt.bar([X for X in range(len(importance))], importance)
plt.show()


# In[13]:


plt.hist(predictions)


# In[14]:


## Best Fit model is RandomForest with better precision and recall


# In[ ]:




