#!/usr/bin/env python
# coding: utf-8

# In[2]:


# IMPORTING IMPORTANT LIBRARIES
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder ,OrdinalEncoder,OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score,RepeatedStratifiedKFold,KFold,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib.inline', '')


# In[3]:


# importing the dataset

data=pd.read_csv('airbnb_listing_train.csv')


# In[4]:


print(data)


# In[5]:


print(data.isnull().sum(
))


# In[6]:


# dropping the neighbourhood_group column 
data=data.drop(['neighbourhood_group'],axis=1)


# In[38]:


print(data.info())


# In[8]:


print(data.isnull().sum(
))


# In[11]:


# dropping the null values
data=data.dropna()


# In[12]:


print(data.shape)


# In[22]:


# creating some visualization
import seaborn as sns
sns.heatmap(data.corr(),annot=True)
plt.show()


# In[23]:


# checking the relationship of data target and differnt features

plt.scatter(data['host_name'],data['price'])
plt.show()


# In[24]:


plt.scatter(data['neighbourhood'],data['price'])
plt.show()


# In[25]:


plt.scatter(data['latitude'],data['price'])
plt.show()


# In[27]:


plt.scatter(data['longitude'],data['price'])
plt.show()


# In[28]:


plt.scatter(data['room_type'],data['price'])
plt.show()


# In[30]:


plt.scatter(data['minimum_nights'],data['price'])
plt.show()


# In[31]:


plt.scatter(data['number_of_reviews'],data['price'])
plt.show()


# In[34]:


plt.scatter(data['last_review'],data['price'])
plt.show()


# In[35]:


plt.scatter(data['reviews_per_month'],data['price'])
plt.show()


# In[36]:


plt.scatter(data['calculated_host_listings_count'],data['price'])
plt.show()


# In[37]:


plt.scatter(data['availability_365'],data['price'])
plt.show()


# In[ ]:





# In[ ]:





# In[14]:


# splittting the data set into features and target

X=data.iloc[:,:-1]
y=data.iloc[:,-1]


# In[15]:


print(X)


# In[16]:


print(y)


# In[18]:


# splitting the data into train and test data

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.33,random_state=15)


# In[20]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[49]:


# categorical and numerical columns for train and test data

X_train_cat=X_train.select_dtypes(include='object')
X_train_num=X_train.select_dtypes(include=['int32','int64','float32','float64'])


X_test_cat=X_test.select_dtypes(include='object')
X_test_num=X_test.select_dtypes(include=['int32','int64','float32','float64'])


# In[50]:


# preprocessing techniques for train and test categorical data


oe=OrdinalEncoder()
oe.fit(X_train_cat)
X_train_cat_enc=oe.transform(X_train_cat)


oe.fit(X_test_cat)
X_test_cat_enc=oe.transform(X_test_cat)




# In[52]:


ss=StandardScaler()
ss.fit(X_train_num)
X_train_num_enc=ss.transform(X_train_num)


ss.fit(X_test_num)
X_test_num_enc=ss.transform(X_test_num)


# In[53]:


# preprocessing for target features train and test data

y_train_df=pd.DataFrame(y_train)
y_test_df=pd.DataFrame(y_test)

ss.fit(y_train_df)
y_train_enc=ss.transform(y_train_df)

ss.fit(y_test_df)
y_test_enc=ss.transform(y_test_df)


# In[54]:


# concat the categorical and numerical data for train and test data
X_train_cat_enc_df=pd.DataFrame(X_train_cat_enc)
X_train_num_enc_df=pd.DataFrame(X_train_num_enc)
X_test_cat_enc_df=pd.DataFrame(X_test_cat_enc)
X_test_num_enc_df=pd.DataFrame(X_test_num_enc)


X_train_final=pd.concat([X_train_cat_enc_df,X_train_num_enc_df],axis=1)
X_test_final=pd.concat([X_test_cat_enc_df,X_test_num_enc_df],axis=1)


# In[55]:


print(X_train_final)


# In[56]:


print(X_test_final)


# In[58]:


# model building

lr=LinearRegression()
lr.fit(X_train_final,y_train_enc)
y_pred=lr.predict(X_test_final)
MSE=mean_squared_error(y_pred,y_test_enc)
print(MSE)


# In[65]:


from sklearn.linear_model import Ridge ,Lasso
RR=Ridge(alpha=.001)
RR.fit(X_train_final,y_train_enc)
y_pred_r=RR.predict(X_test_final)
MSE_R=mean_squared_error(y_pred_r,y_test_enc)
print(MSE_R)


# In[67]:


LR=Lasso(alpha=10.0)
LR.fit(X_train_final,y_train_enc)
y_pred_l=LR.predict(X_test_final)

MAE_L=mean_absolute_error(y_pred_l,y_test_enc)

print(MAE_L)


# In[68]:


from sklearn.ensemble import RandomForestRegressor
RFR=RandomForestRegressor()
RFR.fit(X_train_final,y_train_enc)
y_pred_RFR=RFR.predict(X_test_final)
MSE_RFR=mean_squared_error(y_pred_r,y_test_enc)
print(MSE_RFR)


# In[ ]:




