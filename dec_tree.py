#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np


# In[2]:


df = pd.read_csv("DrugID_Names_Symptoms_Genes_1.csv")
df = df.drop_duplicates()
#df


# In[3]:


epilepy_ID = 'D004827'
epilepy_reflex_ID = 'D020195'


# In[4]:


test_data = df[df.iloc[:,0].str.contains(epilepy_ID)]
affected_rows = test_data.index


# In[5]:


train_data = df.drop(affected_rows)
X_train = train_data.iloc[:,1:-1]
y_train = train_data.iloc[:,-1]

X_test = test_data.iloc[:, 1:-1]
y_test = test_data.iloc[:,-1]


# In[6]:


model = DecisionTreeClassifier()
model = model.fit(X_train, y_train)


# In[7]:


y_pred = model.predict(X_test)


# In[8]:


y_pred


# In[ ]:




