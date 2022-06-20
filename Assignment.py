#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[60]:


data = pd.read_csv('https://raw.githubusercontent.com/regan-mu/ADS-April-2022/main/Assignments/Assignment%202/banking_churn.csv')
data.head()


# In[61]:


data.info()


# In[62]:


print(data.isnull().sum())


# In[63]:


data['Gender'].value_counts()


# In[64]:


data['Gender'].unique()


# In[65]:


data['Geography'].unique()


# In[66]:


data["Gender"] = data["Gender"].astype('category')
data["Geography"] = data["Geography"].astype('category')
data["Surname"] = data["Surname"].astype('category')
data.dtypes


# In[67]:


data.info()


# In[68]:


data = pd.read_csv('https://raw.githubusercontent.com/regan-mu/ADS-April-2022/main/Assignments/Assignment%202/banking_churn.csv')
data = data.drop(['CustomerId','Surname','RowNumber','Geography','Gender'], axis=1)
data.head()


# In[69]:


X = data.drop("Exited", axis=1)
y = data["Exited"]


# In[70]:


X.head()


# In[71]:


y.value_counts()


# In[72]:


get_ipython().system('pip install imbalanced-learn')


# In[73]:


from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy=1)
X_res, y_res = rus.fit_resample(X,y)


# In[74]:


y_res.value_counts()


# In[75]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=100, stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[76]:


y_test.value_counts()


# In[77]:


X_train.head()


# In[78]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)


# In[79]:


clf.get_params()


# In[80]:


#fitting the  model to the data
clf.fit(X_train, y_train)


# In[81]:


#to make prediction
X_test.head()


# In[82]:


y_preds = clf.predict(X_test)
print(y_preds)


# In[83]:


#evaluate the model
clf.score(X_train, y_train)


# In[84]:


clf.score(X_test, y_test)


# In[85]:


#experiment to improve the model
np.random.seed(42)
for i in range(10,151,10):
    print(f"Trying model with {i} estimators...")
    model = RandomForestClassifier(n_estimators=i).fit(X_train,y_train)
    print(f"Model accuracy on test set: {model.score(X_test, y_test) *100}%")
    print("")


# In[86]:


from sklearn.model_selection import cross_val_score
for i in range(10,151,10):
    print(f"Trying model with {i} estimators...")
    model = RandomForestClassifier(n_estimators=i).fit(X_train,y_train)
    print(f"Model accuracy on test set: {model.score(X_test, y_test) *100}%")
    print(f"Cross-calidation score: {np.mean(cross_val_score(model,X,y,cv=5)) * 100}%")
    print("")


# In[87]:


#Grid search cv
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [i for i in range(10,151,10)], 'max_depth': [0,5,10]}

grid = GridSearchCV(RandomForestClassifier(),
                    param_grid,
                    cv=5,
                    scoring='recall'
                   )

grid.fit(X,y)
grid.best_params_
                    


# In[88]:


clf = grid.best_estimator_
clf


# In[89]:


clf = clf.fit(X_train, y_train)


# In[90]:


clf.score(X_test, y_test)


# In[91]:


#Saving the model
import joblib
filename='mymodel'
joblib.dump(model,filename)


# In[92]:


mymodel = joblib.load(filename)


# In[93]:


mymodel.score(X_test, y_test)

